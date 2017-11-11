"""
This module contains the TensorFlow model itself, as well as the logic for training and testing
it in strongly supervised and weakly supervised frameworks.
"""
import sys

sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import pickle
import numpy as np
import os
import sys
import time
import definitions
from seq2seqModel.utils import *
from seq2seqModel.partial_program import *
from data_manager import CNLVRDataSet, DataSetForSupervised, DataSet, load_functions
from seq2seqModel.beam_search import *
from seq2seqModel.hyper_params import *
from general_utils import increment_count, union_dicts
from seq2seqModel.beam_classification import *


def load_meta_data():
    # load word embeddings
    embeddings_file = open(WORD_EMBEDDINGS_PATH, 'rb')
    embeddings_dict = pickle.load(embeddings_file)
    embeddings_file.close()
    assert WORD_EMB_SIZE == np.size(embeddings_dict['blue'])
    vocab_size = len(embeddings_dict)
    vocab_list = [k for k in sorted(embeddings_dict.keys())]
    one_hot_dict = {w: one_hot(vocab_size, i) for i, w in enumerate(vocab_list)}
    embeddings_matrix = np.stack([embeddings_dict[k] for k in vocab_list])

    # load logical tokens inventory
    logical_tokens_mapping = load_functions(LOGICAL_TOKENS_MAPPING_PATH)
    logical_tokens = pickle.load(open(LOGICAL_TOKENS_LIST, 'rb'))
    assert set(logical_tokens) == set(logical_tokens_mapping.keys())

    for var in "xyzwuv":
        logical_tokens.extend([var, 'lambda_{}_:'.format(var)])
    logical_tokens.extend(['<s>', '<EOS>'])
    logical_tokens_ids = {lt: i for i, lt in enumerate(logical_tokens)}
    return logical_tokens_ids, logical_tokens_mapping, embeddings_dict, one_hot_dict, embeddings_matrix


logical_tokens_ids, logical_tokens_mapping, word_embeddings_dict, one_hot_dict, embeddings_matrix = load_meta_data()
if definitions.MANUAL_REPLACEMENTS:
    words_to_tokens = pickle.load(open(os.path.join(definitions.DATA_DIR, 'logical forms', 'words_to_tokens'), 'rb'))
else:
    words_to_tokens = pickle.load(
        open(os.path.join(definitions.DATA_DIR, 'logical forms', 'new_words_to_tokens'), 'rb'))

n_logical_tokens = len(logical_tokens_ids)

if USE_BOW_HISTORY:
    HISTORY_EMB_SIZE += n_logical_tokens


def build_sentence_encoder(vocabulary_size):
    """
    build the computational graph for the lstm sentence encoder. Return only the palceholders and tensors
    that are called from other methods
    """
    sentence_oh_placeholder = tf.placeholder(shape=[None, vocabulary_size], dtype=tf.float32,
                                             name="sentence_placeholder")
    word_embeddings_matrix = tf.get_variable("W_we",  # shape=[vocabulary_size, WORD_EMB_SIZE]
                                             initializer=tf.constant(embeddings_matrix, dtype=tf.float32))
    sentence_embedded = tf.expand_dims(tf.matmul(sentence_oh_placeholder, word_embeddings_matrix), 0)
    # placeholders for sentence and it's length
    sent_lengths = tf.placeholder(dtype=tf.int32, name="sent_length_placeholder")

    # Forward cell
    lstm_fw_cell = BasicLSTMCell(LSTM_HIDDEN_SIZE, forget_bias=1.0)
    # Backward cell
    lstm_bw_cell = BasicLSTMCell(LSTM_HIDDEN_SIZE, forget_bias=1.0)
    # stack cells together in RNN
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, sentence_embedded, sent_lengths,
                                                 dtype=tf.float32)
    #    outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
    #    both output_fw, output_bw will be a `Tensor` shaped: [batch_size, max_time, cell_fw.output_size]`

    # outputs is a (output_forward,output_backwards) tuple. concat them together to receive h vector
    lstm_outputs = tf.concat(outputs, 2)[0]  # shape: [max_time, 2 * hidden_layer_size ]
    final_fw = outputs[0][:, -1, :]
    final_bw = outputs[1][:, 0, :]
    e_m = tf.concat((final_fw, final_bw), axis=1)
    sentence_words_bow = tf.placeholder(tf.float32, [None, len(words_vocabulary)], name="sentence_words_bow")
    e_m_with_bow = tf.concat([e_m, sentence_words_bow], axis=1)

    return sentence_oh_placeholder, sent_lengths, sentence_words_bow, lstm_outputs, e_m_with_bow


def build_decoder(lstm_outputs, final_utterance_embedding):
    """
    build the computational graph for the FF decoder, based on Guu et al. Return only the palceholders and tensors
    that are called from other methods. Names of marics and vectors follow those in the paper.
    """

    history_embedding = tf.placeholder(shape=[None, MAX_DECODING_LENGTH], dtype=tf.float32, name="history_embedding")
    num_rows = tf.shape(history_embedding)[0]
    e_m_tiled = tf.tile(final_utterance_embedding, ([num_rows, 1]))
    decoder_input = tf.concat([e_m_tiled, history_embedding], axis=1)
    W_q = tf.get_variable("W_q", shape=[DECODER_HIDDEN_SIZE, SENT_EMB_SIZE + MAX_DECODING_LENGTH + len(words_vocabulary)],
                          initializer=tf.contrib.layers.xavier_initializer())
    q_t = tf.nn.relu(tf.matmul(W_q, tf.transpose(decoder_input)))
    W_a = tf.get_variable("W_a", shape=[DECODER_HIDDEN_SIZE, SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    alpha = tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(q_t), W_a), tf.transpose(lstm_outputs)))
    c_t = tf.matmul(alpha, lstm_outputs)  # attention vector
    W_s = tf.get_variable("W_s", shape=[HIDDEN_SIZE_NEW, DECODER_HIDDEN_SIZE + SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    W_hidden = tf.get_variable("W_hidden", dtype=tf.float32,
                                       shape=[2, HIDDEN_SIZE_NEW],
                                       initializer=tf.contrib.layers.xavier_initializer())
    token_unnormalized_dist = tf.matmul(W_hidden, tf.matmul(W_s, tf.concat([q_t, tf.transpose(c_t)], 0)))
    return history_embedding, token_unnormalized_dist, W_hidden


def build_batchGrad():
    """
    :return: a list with the placeholders for the gradients of all trained variables in the model
    """
    words_embeddings_grad = tf.placeholder(tf.float32, name="words_embeddings_grad")
    lstm_fw_weights_grad = tf.placeholder(tf.float32, name="lstm_fw_weights_grad")
    lstm_fw_bias_grad = tf.placeholder(tf.float32, name="lstm_fw_bias_grad")
    lstm_bw_weights_grad = tf.placeholder(tf.float32, name="lstm_bw_weights_grad")
    lstm_bw_bias_grad = tf.placeholder(tf.float32, name="lstm_bw_bias_grad")
    wq_grad = tf.placeholder(tf.float32, name="wq_grad")
    wa_grad = tf.placeholder(tf.float32, name="wa_grad")
    ws_grad = tf.placeholder(tf.float32, name="ws_grad")
    logical_tokens_grad = tf.placeholder(tf.float32, name="logical_tokens_grad")
    batchGrad = [words_embeddings_grad, lstm_fw_weights_grad, lstm_fw_bias_grad, lstm_bw_weights_grad,
                 lstm_bw_bias_grad,
                 wq_grad, wa_grad, ws_grad, logical_tokens_grad]
    return batchGrad




def get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, sentence_words_bow,
                                 encoder_output_tensors, learn_embeddings=False):
    """
    creates the values needed and feed-dicts that depend on the sentence.
    these feed dicts are used to run or to compute gradients.
    """

    sentence_matrix = np.stack([one_hot_dict.get(w, one_hot_dict['<UNK>']) for w in sentence.split()])
    bow_words = np.reshape(np.sum([words_array == x for x in sentence.split()], axis=0), [1, len(words_vocabulary)])

    length = [len(sentence.split())]
    encoder_feed_dict = {sentence_placeholder: sentence_matrix, sent_lengths_placeholder: length,
                         sentence_words_bow: bow_words}
    sentence_encoder_outputs = sess.run(encoder_output_tensors, feed_dict=encoder_feed_dict)
    decoder_feed_dict = {encoder_output_tensors[i]: sentence_encoder_outputs[i]
                         for i in range(len(encoder_output_tensors))}

    if not learn_embeddings:
        W_we = tf.get_default_graph().get_tensor_by_name('W_we:0')
        encoder_feed_dict = union_dicts(encoder_feed_dict, {W_we: embeddings_matrix})
    return encoder_feed_dict, decoder_feed_dict


def get_feed_dicts_from_program(program, logical_tokens_embeddings_dict, program_dependent_placeholders,
                                skipped_indices=[]):
    """
    creates the values needed and feed-dicts that depend on the program. (these include the history embeddings
    and the chosen tokens, represented as one-hot vectors)
    both used for computing gradients and the the first used also in every forward run.
    """

    histories = []
    tokens_one_hot = []
    for i in range(len(program)):
        if i in skipped_indices:
            continue
        history_tokens = ['<s>' for _ in range(MAX_DECODING_LENGTH - i)] + \
                         program[max(0, i - MAX_DECODING_LENGTH): i]

        BOW_history = [sparse_vector_from_indices(n_logical_tokens,
                                                  [logical_tokens_ids[tok] for tok in program])] \
            if USE_BOW_HISTORY else []
        history_embs = [logical_tokens_embeddings_dict[tok] for tok in history_tokens] + BOW_history
        history_embs = np.reshape(np.concatenate(history_embs), [1, MAX_DECODING_LENGTH])
        histories.append(history_embs)
        tokens_one_hot.append(one_hot(n_logical_tokens, logical_tokens_ids[program[i]]))

    one_hot_stacked = np.stack(tokens_one_hot)
    histories_stacked = np.squeeze(np.stack(histories), axis=1)
    return {
        program_dependent_placeholders[0]: histories_stacked,
        program_dependent_placeholders[1]: one_hot_stacked
    }


# build the computaional graph:
# bi-lstm encoder - given a sentence (of a variable length) as a sequence of word embeddings,
# and returns the lstm outputs.

sentence_placeholder, sent_lengths_placeholder, sentence_words_bow, h, e_m = build_sentence_encoder(
    vocabulary_size=len(one_hot_dict))

# ff decoder - given the outputs of the encoder, and an embedding of the decoding history,
# computes a probability distribution over the tokens.
history_embedding_placeholder, token_unnormalized_dist, W_logical_tokens = build_decoder(h, e_m)

#not used
valid_logical_tokens = tf.placeholder(tf.float32, [None, n_logical_tokens],  ##
                                      name="valid_logical_tokens")
token_prob_dist_tensor = tf.nn.softmax(token_unnormalized_dist, dim=0)
chosen_logical_tokens = tf.placeholder(tf.float32, [None, n_logical_tokens],  ##
                                       name="chosen_logical_tokens")  # a one-hot vector represents the action taken at each step
invalid_logical_tokens_mask = tf.placeholder(tf.float32, [None, n_logical_tokens],  ##
                                             name="invalid_logical_tokens_mask")

program_dependent_placeholders = (history_embedding_placeholder,
                                  chosen_logical_tokens,
                                  invalid_logical_tokens_mask)

logits = tf.transpose(token_unnormalized_dist)  # + invalid_logical_tokens_mask

labels_placeholder = tf.placeholder(tf.float32, shape=(BEAM_SIZE,1))
logits = tf.reshape(logits,(1,BEAM_SIZE))
labels_cur = tf.reshape(labels_placeholder,(1,BEAM_SIZE))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels_cur, logits=logits, name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')


def run_model(sess, dataset, mode, validation_dataset=None, load_params_path=None, save_model_path=None,
              return_sentences_results=None, beam_classifier=False, beam_classifier_test=False, clf_params_path=None,
              beam_reranking_train=False):


    modes = ('train', 'test')
    assert mode in modes
    test_between_training_epochs = validation_dataset is not None
    if (mode == 'test'):
        test_between_training_epochs = False

    # initialization
    theta = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    compute_program_grads = optimizer.compute_gradients(cross_entropy)
    batch_grad = build_batchGrad()
    update_grads = optimizer.apply_gradients(zip(batch_grad, theta))
    sess.run(tf.global_variables_initializer())

    # load pre-trained variables, if given
    if load_params_path:
        tf.train.Saver(var_list=theta).restore(sess, load_params_path)
    if save_model_path:
        saver2 = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    # initialize gradient buffers to zero
    gradList = sess.run(theta)
    gradBuffer = {}
    for i, grad in enumerate(gradList):
        gradBuffer[i] = grad * 0


    curr_dataset, other_dataset = dataset, validation_dataset
    curr_mode = mode
    epochs = 0

    while epochs < 30:

        # get a mini-batch of sentences and their related images
        batch, is_last_batch_in_epoch = curr_dataset.next_batch(BATCH_SIZE_UNSUPERVISED)
        batch_sentence_ids = [key for key in batch.keys()]
        sentences, samples = zip(*[batch[k] for k in batch_sentence_ids])

        # get a dictionary of the current embeddings of the logical tokens
        # (in train they are change after every mini batch - as they are also trainable variables)
        current_logical_tokens_embeddings = sess.run(W_logical_tokens)
        logical_tokens_embeddings_dict = \
            {token: current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens_ids}

        sum_loss = 0
        total = 0
        epochs += 1

        for step in range(len(sentences)):

            sentence_id = batch_sentence_ids[step]
            sentence = (sentences[step])
            related_samples = samples[step]
            program = programs[step]
            label = labels[step]

            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder,
                                             sentence_words_bow, (h, e_m), LEARN_EMBEDDINGS)

            program_dependent_feed_dict = get_feed_dicts_from_program(
                program, logical_tokens_embeddings_dict, program_dependent_placeholders)
            program_grad, ce = sess.run([compute_program_grads,loss], feed_dict=union_dicts(encoder_feed_dict, program_dependent_feed_dict))
            for i, grad in enumerate(program_grad):
                gradBuffer[i] += grad[0]
            sum_loss += ce
            total += 1

            if step % BATCH_SIZE_UNSUPERVISED == 0:
                sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})
                print("average loss in batch:",sum_loss/total)
                for i, grad in enumerate(gradBuffer):
                    gradBuffer[i] = gradBuffer[i] * 0
                sum_loss = 0
                total = 0



    return {}


