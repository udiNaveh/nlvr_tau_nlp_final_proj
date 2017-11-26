import sys

sys.path.append('../')
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import pickle
import definitions
from seq2seqModel.utils import *
from seq2seqModel.partial_program import *
from data_manager import CNLVRDataSet, DataSetForSupervised, DataSet, load_functions
from seq2seqModel.beam_search import *
from seq2seqModel.hyper_params import *
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





def build_sentence_encoder(vocabulary_size, embeddings_matrix):
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

    history_embedding = tf.placeholder(shape=[None, MAX_DECODING_LENGTH*LOG_TOKEN_EMB_SIZE], dtype=tf.float32, name="history_embedding")
    num_rows = tf.shape(history_embedding)[0]
    e_m_tiled = tf.tile(final_utterance_embedding, ([num_rows, 1]))
    decoder_input = tf.concat([e_m_tiled, history_embedding], axis=1)
    W_q = tf.get_variable("W_q", shape=[DECODER_HIDDEN_SIZE_NEW, SENT_EMB_SIZE + MAX_DECODING_LENGTH*LOG_TOKEN_EMB_SIZE + len(words_vocabulary)],
                          initializer=tf.contrib.layers.xavier_initializer())
    q_t = tf.nn.relu(tf.matmul(W_q, tf.transpose(decoder_input)))
    W_a = tf.get_variable("W_a", shape=[DECODER_HIDDEN_SIZE_NEW, SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    alpha = tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(q_t), W_a), tf.transpose(lstm_outputs)))
    c_t = tf.matmul(alpha, lstm_outputs)  # attention vector
    W_s = tf.get_variable("W_s", shape=[DECODER_HIDDEN_SIZE_NEW, DECODER_HIDDEN_SIZE_NEW + SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    W_hidden = tf.get_variable("W_hidden", dtype=tf.float32,
                                       shape=[2, DECODER_HIDDEN_SIZE_NEW],
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



def run_similarity_model(sess, examples, current_logical_tokens_embeddings, load_params_path=None, save_model_path=None,
              inference=False, reuse=False):

    tf.set_random_seed(1)
    np.random.seed(1)
    # build the computaional graph:

    with tf.variable_scope("similarity", reuse=reuse):
        logical_tokens_ids, logical_tokens_mapping, word_embeddings_dict, one_hot_dict, embeddings_matrix = load_meta_data()
        sentence_placeholder, sent_lengths_placeholder, sentence_words_bow, h, e_m = build_sentence_encoder(
            len(one_hot_dict), embeddings_matrix)

        history_embedding_placeholder, token_unnormalized_dist, W_hidden = build_decoder(h, e_m)

        logits = tf.transpose(token_unnormalized_dist)

        labels_placeholder = tf.placeholder(tf.float32, shape=(2,1))
        logits = tf.reshape(logits, (1, 2))
        labels_cur = tf.reshape(labels_placeholder, (1, 2))
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_cur, logits=logits, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

        # initialization
        theta = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        compute_program_grads = optimizer.compute_gradients(cross_entropy)#TODO loss?
        batch_grad = build_batchGrad()
        update_grads = optimizer.apply_gradients(zip(batch_grad, theta))
        sess.run(tf.global_variables_initializer())

        # load pre-trained variables, if given
        if load_params_path:
            tf.train.Saver(var_list=theta).restore(sess, load_params_path)
        if save_model_path:
            saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

        # initialize gradient buffers to zero
        gradList = sess.run(theta)
        gradBuffer = {}
        for i, grad in enumerate(gradList):
            gradBuffer[i] = grad * 0

        epochs = 0

        #examples = pickle.load(open(examples_file, 'rb'))
        #current_logical_tokens_embeddings = pickle.load(open(logical_tokens_embeddings_file,'rb'))#TODO can also initialize word embeddings
        logical_tokens_embeddings_dict = \
            {token: current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens_ids}

        while epochs < 30:

            np.random.shuffle(examples)

            sum_loss = 0
            total = 0
            epochs += 1

            for step in range(len(examples)):

                sentence = (examples[step][0])
                program = examples[step][1]
                label = examples[step][2]

                label_one_hot = np.zeros((2,1))
                label_one_hot[int(label)][0] = 1
                if np.sum(label_one_hot)!= 1:
                    print("invalid label")

                sentence_matrix = np.stack([one_hot_dict.get(w, one_hot_dict['<UNK>']) for w in sentence.split()])
                bow_words = np.reshape(np.sum([words_array == x for x in sentence.split()], axis=0),
                                       [1, len(words_vocabulary)])

                length = [len(sentence.split())]

                history_tokens = ['<s>' for _ in range(MAX_DECODING_LENGTH - len(program))] + \
                                 program[-MAX_DECODING_LENGTH:]

                history_embs = [logical_tokens_embeddings_dict[tok] for tok in history_tokens]
                history_embs = np.reshape(np.concatenate(history_embs), [1, MAX_DECODING_LENGTH*LOG_TOKEN_EMB_SIZE])

                feed_dict = {sentence_placeholder: sentence_matrix, sent_lengths_placeholder: length,
                             sentence_words_bow: bow_words, history_embedding_placeholder: history_embs,
                             labels_placeholder: label_one_hot}

                if inference:
                    # if the model is used in inference, it returns the probability of the sentence and the program to be consistent
                    result = tf.nn.softmax(sess.run(logits, feed_dict=feed_dict)).eval()
                    return result[0,1]

                program_grad, ce = sess.run([compute_program_grads,loss], feed_dict=feed_dict)
                for i, grad in enumerate(program_grad):
                    gradBuffer[i] += grad[0]
                sum_loss += ce
                total += 1

                if step % BATCH_SIZE_UNSUPERVISED == 0:
                    sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})

                    for i, grad in enumerate(gradBuffer):
                        gradBuffer[i] = gradBuffer[i] * 0
                    if step % (BATCH_SIZE_UNSUPERVISED*10) ==0:
                        print("epoch: ", epochs, "average loss in epoch so far:", sum_loss / total)
                        #sum_loss = 0
                        #total = 0
    time_stamp = time.strftime("%Y-%m-%d_%H_%M")
    if save_model_path:
        saver.save(sess,save_model_path)

    return {}


