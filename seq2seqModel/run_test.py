import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from seq2seqModel.utils import *
from handle_data import CNLVRDataSet, SupervisedParsing
import pickle
import numpy as np
import time
import os
import definitions
from seq2seqModel.beam import *
import time

#paths

LOGICAL_TOKENS_MAPPING_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping_limitations')
WORD_EMBEDDINGS_PATH = os.path.join(definitions.ROOT_DIR, 'word2vec', 'embeddings_10iters_12dim')
PARSED_EXAMPLES_T = os.path.join(definitions.DATA_DIR, 'parsed sentences', 'parses for check as tokens')
TRAINED_WEIGHTS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables.ckpt')
TRAINED_WEIGHTS2 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables2.ckpt')
TRAINED_WEIGHTS3 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables3.ckpt')
SENTENCES_IN_PRETRAIN_PATTERNS = os.path.join(definitions.DATA_DIR, 'parsed sentences', 'sentences_in_pattern')
LOGICAL_TOKENS_LIST =  os.path.join(definitions.DATA_DIR, 'logical forms', 'logical_tokens_list')

####
###hyperparameters
####

#dimensions
WORD_EMB_SIZE = 12
LOG_TOKEN_EMB_SIZE = 12
DECODER_HIDDEN_SIZE = 50
LSTM_HIDDEN_SIZE = 30
SENT_EMB_SIZE = 2 * LSTM_HIDDEN_SIZE
HISTORY_LENGTH = 4

#other hyper parameters
LEARNING_RATE = 0.001
BETA = 1

BATCH_SIZE_UNSUPERVISED = 8
BATCH_SIZE_SUPERVISED = 10
USE_BOW_HISTORY = False
IRRELEVANT_TOKENS_IN_GRAD = True
AUTOMATIC_TOKENS_IN_GRAD = False
HISTORY_EMB_SIZE = HISTORY_LENGTH * LOG_TOKEN_EMB_SIZE




# load word embeddings
embeddings_file = open(WORD_EMBEDDINGS_PATH,'rb')
embeddings_dict = pickle.load(embeddings_file)
embeddings_file.close()

def load_meta_data():

    # load word embeddings
    embeddings_file = open(WORD_EMBEDDINGS_PATH,'rb')
    embeddings_dict = pickle.load(embeddings_file)
    embeddings_file.close()
    assert WORD_EMB_SIZE == np.size(embeddings_dict['blue'])

    # load logical tokens inventory
    logical_tokens_mapping = load_functions(LOGICAL_TOKENS_MAPPING_PATH)
    logical_tokens = pickle.load(open(LOGICAL_TOKENS_LIST,'rb'))
    assert set(logical_tokens) == set(logical_tokens_mapping.keys())

    for var in "xyzwuv":
        logical_tokens.extend([var, 'lambda_{}_:'.format(var) ])
    logical_tokens.extend(['<s>', '<EOS>'])
    logical_tokens_ids = {lt: i for i, lt in enumerate(logical_tokens)}
    return  logical_tokens_ids, logical_tokens_mapping, embeddings_dict


logical_tokens_ids, logical_tokens_mapping, word_embeddings_dict = load_meta_data()
n_logical_tokens = len(logical_tokens_ids)

if USE_BOW_HISTORY:
    HISTORY_EMB_SIZE += n_logical_tokens

def build_sentence_encoder():

    # placeholders for sentence and it's length
    sentence_placeholder = tf.placeholder(shape = [None, None, WORD_EMB_SIZE], dtype = tf.float32, name ="sentence_placeholder")
    sent_lengths = tf.placeholder(dtype = tf.int32,name = "sent_length_placeholder")

    # Forward cell
    lstm_fw_cell = BasicLSTMCell(LSTM_HIDDEN_SIZE, forget_bias=1.0)
    # Backward cell
    lstm_bw_cell = BasicLSTMCell(LSTM_HIDDEN_SIZE, forget_bias=1.0)
    # stack cells together in RNN
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, sentence_placeholder,sent_lengths,dtype=tf.float32)
    #    outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
    #    both output_fw, output_bw will be a `Tensor` shaped: [batch_size, max_time, cell_fw.output_size]`

    # outputs is a (output_forward,output_backwards) tuple. concat them together to receive h vector
    lstm_outputs = tf.concat(outputs,2)[0]    # shape: [batch_size, max_time, 2 * hidden_layer_size ]
    # the final utterance is the last output

    final_fw = outputs[0][:,-1,:]
    final_bw = outputs[1][:,0,:]

    return sentence_placeholder, sent_lengths, lstm_outputs, tf.concat((final_fw, final_bw), axis=1)


def build_decoder(lstm_outputs, final_utterance_embedding):
    history_embedding = tf.placeholder(shape=[None, HISTORY_EMB_SIZE], dtype=tf.float32, name="history_embedding")
    num_rows = tf.shape(history_embedding)[0]
    e_m_tiled = tf.tile(final_utterance_embedding, ([num_rows, 1]))
    decoder_input = tf.concat([e_m_tiled, history_embedding], axis=1)
    W_q = tf.get_variable("W_q", shape=[DECODER_HIDDEN_SIZE, SENT_EMB_SIZE + HISTORY_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())  # (d1,100)
    q_t = tf.nn.relu(tf.matmul(W_q, tf.transpose(decoder_input)))  # dim [d1,1]
    W_a = tf.get_variable("W_a", shape=[DECODER_HIDDEN_SIZE, SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())  # dim [d1*60]
    alpha = tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(q_t), W_a), tf.transpose(lstm_outputs)))  # dim [1,25]
    c_t = tf.matmul(alpha, lstm_outputs)  # attention vector, dim [1,60]
    W_s = tf.get_variable("W_s", shape=[LOG_TOKEN_EMB_SIZE, DECODER_HIDDEN_SIZE + SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    W_logical_tokens = tf.get_variable("W_logical_tokens", dtype= tf.float32,
                                       shape=[n_logical_tokens, LOG_TOKEN_EMB_SIZE],
                                       initializer=tf.contrib.layers.xavier_initializer())
    token_unnormalized_dist = tf.matmul(W_logical_tokens, tf.matmul(W_s, tf.concat([q_t, tf.transpose(c_t)], 0)))

    return history_embedding, token_unnormalized_dist, W_logical_tokens

# PartialProgram


def sample_valid_decodings(next_token_probs_getter, n_decodings):
    decodings = []
    while len(decodings)<n_decodings:
        partial_program = PartialProgram()
        for t in range(max_decoding_length+1):
            if t > 0 and partial_program[-1] == '<EOS>':
                decodings.append(partial_program)
                break
            valid_next_tokens, probs_given_valid = \
                next_token_probs_getter(partial_program)
            if not valid_next_tokens:
                break
            next_token = np.random.choice(valid_next_tokens, p= probs_given_valid)
            p = probs_given_valid[valid_next_tokens.index(next_token)]
            partial_program.add_token(next_token, np.log(p), logical_tokens_mapping)
    return decodings


def sample_decoding_prefixes(next_token_probs_getter, n_decodings, length):
    decodings = []
    while len(decodings)<n_decodings:
        partial_program = PartialProgram()
        for t in range(length):
            if t > 0 and partial_program[-1] == '<EOS>':
                decodings.append(partial_program)
                break
            valid_next_tokens, probs_given_valid = \
                next_token_probs_getter(partial_program)
            if not valid_next_tokens:
                break
            next_token = np.random.choice(valid_next_tokens, p= probs_given_valid)
            p = probs_given_valid[valid_next_tokens.index(next_token)]
            partial_program.add_token(next_token, np.log(p), logical_tokens_mapping)
        decodings.append(partial_program)
    return decodings

def create_partial_program(next_token_probs_getter, token_seq):
    partial_program = PartialProgram()
    for tok in token_seq:
        valid_next_tokens, probs_given_valid = \
            next_token_probs_getter(partial_program)
        if tok not in valid_next_tokens:
            return None
        p = probs_given_valid[valid_next_tokens.index(tok)]
        partial_program.add_token(tok, np.log(p), logical_tokens_mapping)
    return partial_program




def get_next_token_probs(sess, partial_program, logical_tokens_embeddings_dict, decoder_feed_dict, history_embedding_tensor,
                         token_prob_dist):
    valid_next_tokens = partial_program.get_possible_continuations()
    if len(valid_next_tokens) == 1:
        return valid_next_tokens, [1.0]
    history_tokens = ['<s>' for _ in range(HISTORY_LENGTH - len(partial_program))] + \
                     partial_program[-HISTORY_LENGTH:]

    BOW_history = [sparse_vector_from_indices(n_logical_tokens, [logical_tokens_ids[tok] for tok in partial_program])] \
        if USE_BOW_HISTORY else []

    history_embs = [logical_tokens_embeddings_dict[tok] for tok in history_tokens] + BOW_history
    history_embs = np.reshape(np.concatenate(history_embs), [1, HISTORY_EMB_SIZE])

    # run forward pass
    decoder_feed_dict[history_embedding_tensor] = history_embs
    current_probs = np.squeeze(sess.run(token_prob_dist, feed_dict=decoder_feed_dict))
    current_probs = np.where(current_probs>0, current_probs, 1e-30)
    if (np.count_nonzero(current_probs) != len(current_probs)):
        print("zero prob")

    probs_given_valid = [current_probs[logical_tokens_ids[next_tok]] for next_tok in valid_next_tokens]

    probs_given_valid = probs_given_valid / np.sum(probs_given_valid)
    return valid_next_tokens, probs_given_valid


def get_gradient_weights_for_beam(beam_rewarded_programs):

    beam_log_probs = np.array([prog.logprob for prog in beam_rewarded_programs])
    q_mml = softmax(beam_log_probs)
    return np.power(q_mml, BETA) / np.sum(np.power(q_mml, BETA))


def sentences_to_embeddings(sentences, embeddings_dict):
    return np.array([[embeddings_dict.get(w, embeddings_dict['<UNK>']) for w in sentence] for sentence in sentences])



def get_feed_dicts_from_sentence(sess, sentence, sentence_placeholder, sent_lengths_placeholder, encoder_output_tensors):

    sentence_embedding = np.reshape([word_embeddings_dict.get(w, word_embeddings_dict['<UNK>']) for w in sentence.split()],
                                    [1, len(sentence.split()), WORD_EMB_SIZE])
    length = [len(sentence.split())]
    encoder_feed_dict = {sentence_placeholder: sentence_embedding, sent_lengths_placeholder: length}
    sentence_encoder_outputs = sess.run(encoder_output_tensors, feed_dict= encoder_feed_dict)
    decoder_feed_dict = {encoder_output_tensors[i] : sentence_encoder_outputs[i]
                         for i in range(len(encoder_output_tensors))}
    return encoder_feed_dict, decoder_feed_dict


def get_feed_dicts_from_program(program, logical_tokens_embeddings_dict, program_dependent_placeholders,
                                skipped_indices = []):
    histories = []
    tokens_one_hot = []
    for i in range(len(program)):
        if i in skipped_indices:
            continue
        history_tokens = ['<s>' for _ in range(HISTORY_LENGTH - i)] + \
                         program[max(0, i - HISTORY_LENGTH): i]

        BOW_history = [sparse_vector_from_indices(n_logical_tokens,
                                                  [logical_tokens_ids[tok] for tok in program])] \
            if USE_BOW_HISTORY else []
        history_embs = [logical_tokens_embeddings_dict[tok] for tok in history_tokens] + BOW_history
        history_embs = np.reshape(np.concatenate(history_embs), [1, HISTORY_EMB_SIZE])
        histories.append(history_embs)
        tokens_one_hot.append(one_hot(n_logical_tokens, logical_tokens_ids[program[i]]))

    one_hot_stacked = np.stack(tokens_one_hot)
    histories_stacked = np.squeeze(np.stack(histories), axis=1)
    return {
            program_dependent_placeholders[0] : histories_stacked,
            program_dependent_placeholders[1] : one_hot_stacked
    }



def run_unsupervised_inference(sess,data, load_params):
    # build the computaional graph:
    # bi-lstm encoder - given a sentence (of a variable length) as a sequence of word embeddings,
    # and returns the lstm outputs.
    sentence_placeholder, sent_lengths_placeholder, h, e_m = build_sentence_encoder()
    # ff decoder - given the outputs of the encoder, and an embedding of the decoding history,
    # computes a probability distribution over the tokens.
    history_embedding_placeholder, token_unnormalized_dist, W_logical_tokens = build_decoder(h, e_m)
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

    # cross-entropy loss per single token in a single sentence
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=chosen_logical_tokens, logits=logits))

    theta = tf.trainable_variables()

    init = tf.global_variables_initializer()
    sess.run(init)

    if load_params:
        tf.train.Saver(theta).restore(sess, load_params)


    current_logical_tokens_embeddings = sess.run(W_logical_tokens)
    logical_tokens_embeddings_dict = \
        {token: current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens_ids}

    empty_beam = 0
    batch_size = 1
    correct_avg = 0
    correct_first = 0
    total = 0
    batch_num = 1
    total_correct = 0
    correct_beam_parses = open("correct beams 8.txt ", 'w')
    num_consistent_per_sentence = []
    beam_final_sizes = []
    accuracy_by_prog_rank = {}
    n_images = 0
    iter = 0
    samples_num = len(data.samples.values())

    for sample in data.samples.values():
        if total % 10 == 0:
            print("sample %d out of %d" % (total,samples_num))

        sentences = [sample.sentence]
        label = sample.label

        embedded_sentences = sentences_to_embeddings(sentences, embeddings_dict)

        for step in range(batch_size):

            s = (sentences[step])
            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sess,s, sentence_placeholder, sent_lengths_placeholder, (h, e_m))

            next_token_probs_getter = lambda pp :  get_next_token_probs(sess,pp, logical_tokens_embeddings_dict,
                                                                        decoder_feed_dict,
                                                                        history_embedding_placeholder,
                                                                        token_prob_dist_tensor)

            beam = e_greedy_randomized_beam_search(next_token_probs_getter, logical_tokens_mapping,
                                                   original_sentence= s)

            rewarded_programs = []
            compiled = 0
            correct = 0

            execution_results = []
            for prog in beam:
                prog.token_seq.pop(-1) # take out the '<EOS>' token
                execution_results.append(execute(prog.token_seq,sample.structured_rep,logical_tokens_mapping))

            if execution_results[0] is None:
                nan_count+=1
            execution_results = [x for x in execution_results if x is not None]
            if len(execution_results) == 0:
                execution_results.append(1)
                empty_beam += 1
            avg = np.array(execution_results).mean()
            avg = 1 if avg >= 0.5 else 0
            correct_avg += 1 if (avg == label) else 0
            correct_first += 1 if execution_results[0] == label else 0
            total += 1


        if total % 50 == 0:
            print("accuracy for average of beam: %.2f" % (correct_avg / total))
            print("accuracy for largest p in beam: %.2f" % (correct_first / total))
            print("empty beam and None cases proportion: %.2f" % (empty_beam / total))
    print("total accuracy for average of beam: %.2f" % (correct_avg / total))
    print("total accuracy for largest p in beam: %.2f" % (correct_first / total))
    print("total empty beam cases proportion: %.2f" % (empty_beam / total))




TRAINED_WEIGHTS_SUPERVISED = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables2.ckpt')
TRAINED_WEIGHT_UNSUPERVISED = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables3.ckpt')
TRAINED_WEIGHTS4 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables4.ckpt')


if __name__ == '__main__':

    data = CNLVRDataSet(definitions.TRAIN_JSON, ignore_all_true=False)
    start = time.time()
    with tf.Session() as sess:
        print("running with pre-trained unsupervised")
        run_unsupervised_inference(sess,data, load_params=TRAINED_WEIGHTS_SUPERVISED)#TRAINED_WEIGHTS_UNS
        finish = time.time()
        print("elapsed time: %s" % (time.strftime("%H%M%S",time.localtime(finish-start))))
        print("running without pre-training unsupervised:")
    #with tf.Session as sess:
        start = time.time()
        run_unsupervised_inference(sess,data, load_params=TRAINED_WEIGHTS_SUPERVISED)#TRAINED_WEIGHTS2_SUP
        finish = time.time()
        print("elapsed time: %s" % (time.strftime("%H%M%S", time.localtime(finish - start))))