"""
This module contains the TensorFlow model itself, as well as the logic for training and testing
it in strongly supervised and weakly supervised frameworks.
"""

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
    embeddings_file = open(WORD_EMBEDDINGS_PATH,'rb')
    embeddings_dict = pickle.load(embeddings_file)
    embeddings_file.close()
    assert WORD_EMB_SIZE == np.size(embeddings_dict['blue'])
    vocab_size = len(embeddings_dict)
    vocab_list = [k for k in sorted(embeddings_dict.keys())]
    one_hot_dict = {w : one_hot(vocab_size, i) for i,w in enumerate(vocab_list)}
    embeddings_matrix = np.stack([embeddings_dict[k] for k in vocab_list])

    # load logical tokens inventory
    logical_tokens_mapping = load_functions(LOGICAL_TOKENS_MAPPING_PATH)
    logical_tokens = pickle.load(open(LOGICAL_TOKENS_LIST,'rb'))
    assert set(logical_tokens) == set(logical_tokens_mapping.keys())

    for var in "xyzwuv":
        logical_tokens.extend([var, 'lambda_{}_:'.format(var) ])
    logical_tokens.extend(['<s>', '<EOS>'])
    logical_tokens_ids = {lt: i for i, lt in enumerate(logical_tokens)}
    return  logical_tokens_ids, logical_tokens_mapping, embeddings_dict, one_hot_dict, embeddings_matrix

logical_tokens_ids, logical_tokens_mapping, word_embeddings_dict, one_hot_dict, embeddings_matrix = load_meta_data()
words_to_tokens = pickle.load(open(os.path.join(definitions.DATA_DIR, 'logical forms', 'words_to_tokens'), 'rb'))



n_logical_tokens = len(logical_tokens_ids)

if USE_BOW_HISTORY:
    HISTORY_EMB_SIZE += n_logical_tokens


def build_sentence_encoder(vocabulary_size):
    """
    build the computational graph for the lstm sentence encoder. Return only the palceholders and tensors
    that are called from other methods
    """
    sentence_oh_placeholder = tf.placeholder(shape=[None, vocabulary_size], dtype=tf.float32, name="sentence_placeholder")
    word_embeddings_matrix = tf.get_variable("W_we", #shape=[vocabulary_size, WORD_EMB_SIZE]
                                             initializer=tf.constant(embeddings_matrix, dtype=tf.float32))
    sentence_embedded = tf.expand_dims(tf.matmul(sentence_oh_placeholder, word_embeddings_matrix), 0)
    # placeholders for sentence and it's length
    sent_lengths = tf.placeholder(dtype = tf.int32,name = "sent_length_placeholder")

    # Forward cell
    lstm_fw_cell = BasicLSTMCell (LSTM_HIDDEN_SIZE, forget_bias=1.0)
    # Backward cell
    lstm_bw_cell = BasicLSTMCell(LSTM_HIDDEN_SIZE, forget_bias=1.0)
    # stack cells together in RNN
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, sentence_embedded,sent_lengths,dtype=tf.float32)
    #    outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output `Tensor`.
    #    both output_fw, output_bw will be a `Tensor` shaped: [batch_size, max_time, cell_fw.output_size]`

    # outputs is a (output_forward,output_backwards) tuple. concat them together to receive h vector
    lstm_outputs = tf.concat(outputs,2)[0]    # shape: [max_time, 2 * hidden_layer_size ]
    final_fw = outputs[0][:,-1,:]
    final_bw = outputs[1][:,0,:]

    return sentence_oh_placeholder, sent_lengths, lstm_outputs, tf.concat((final_fw, final_bw), axis=1)




def build_decoder(lstm_outputs, final_utterance_embedding):
    """
    build the computational graph for the FF decoder, based on Guu et al. Return only the palceholders and tensors
    that are called from other methods. Names of marics and vectors follow those in the paper.
    """

    history_embedding = tf.placeholder(shape=[None, HISTORY_EMB_SIZE], dtype=tf.float32, name="history_embedding")
    num_rows = tf.shape(history_embedding)[0]
    e_m_tiled = tf.tile(final_utterance_embedding, ([num_rows, 1]))
    decoder_input = tf.concat([e_m_tiled, history_embedding], axis=1)
    W_q = tf.get_variable("W_q", shape=[DECODER_HIDDEN_SIZE, SENT_EMB_SIZE + HISTORY_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    q_t = tf.nn.relu(tf.matmul(W_q, tf.transpose(decoder_input)))
    W_a = tf.get_variable("W_a", shape=[DECODER_HIDDEN_SIZE, SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    alpha = tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(q_t), W_a), tf.transpose(lstm_outputs)))
    c_t = tf.matmul(alpha, lstm_outputs)  # attention vector
    W_s = tf.get_variable("W_s", shape=[LOG_TOKEN_EMB_SIZE, DECODER_HIDDEN_SIZE + SENT_EMB_SIZE],
                          initializer=tf.contrib.layers.xavier_initializer())
    W_logical_tokens = tf.get_variable("W_logical_tokens", dtype= tf.float32,
                                       shape=[n_logical_tokens, LOG_TOKEN_EMB_SIZE],
                                       initializer=tf.contrib.layers.xavier_initializer())
    token_unnormalized_dist = tf.matmul(W_logical_tokens, tf.matmul(W_s, tf.concat([q_t, tf.transpose(c_t)], 0)))
    return history_embedding, token_unnormalized_dist, W_logical_tokens


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
    batchGrad = [words_embeddings_grad, lstm_fw_weights_grad, lstm_fw_bias_grad, lstm_bw_weights_grad, lstm_bw_bias_grad,
                 wq_grad, wa_grad, ws_grad, logical_tokens_grad]
    return batchGrad


def get_next_token_probs_from_nn(partial_program, logical_tokens_embeddings_dict, decoder_feed_dict,
                                 history_embedding_placeholder, token_prob_dist_tensor):
    """
    Uses the current state of a partial program to get the list of candidates for the next token in the program,
    along with their probabilities according to the current weights of the model.
    """

    valid_next_tokens = partial_program.get_possible_continuations()

    if len(valid_next_tokens)==1:
        return valid_next_tokens, [1.0]

    history_tokens = ['<s>' for _ in range(HISTORY_LENGTH - len(partial_program))] + \
                     partial_program[-HISTORY_LENGTH:]

    BOW_history = [sparse_vector_from_indices(n_logical_tokens, [logical_tokens_ids[tok] for tok in partial_program])] \
        if USE_BOW_HISTORY else []

    history_embs = [logical_tokens_embeddings_dict[tok] for tok in history_tokens] + BOW_history
    history_embs = np.reshape(np.concatenate(history_embs), [1, HISTORY_EMB_SIZE])

    # run forward pass
    decoder_feed_dict[history_embedding_placeholder] = history_embs
    current_probs = np.squeeze(sess.run(token_prob_dist_tensor, feed_dict=decoder_feed_dict))

    # change zero probs to almost-zero to keep numerical stability
    current_probs = np.where(current_probs>0, current_probs, 1e-30)

    probs_given_valid = [1.0] if len(valid_next_tokens) == 1 else \
        [current_probs[logical_tokens_ids[next_tok]] for next_tok in valid_next_tokens]
    probs_given_valid = probs_given_valid / np.sum(probs_given_valid)
    return valid_next_tokens, probs_given_valid


def get_gradient_weights_for_programs(beam_rewarded_programs):
    """
    :param beam_rewarded_programs: a list of all programs that receive rewards
    :return: the weights vector for the programs gradients using the beta-meritocratic approach
    """
    if not beam_rewarded_programs:
        return []
    beam_log_probs = np.array([prog.logprob for prog in beam_rewarded_programs])
    q_mml = softmax(beam_log_probs)
    return np.power(q_mml, BETA) / np.sum(np.power(q_mml, BETA))


def get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, encoder_output_tensors, learn_embeddings=False):
    """
    creates the values needed and feed-dicts that depend on the sentence.
    these feed dicts are used to run or to compute gradients.
    """

    sentence_matrix = np.stack([one_hot_dict.get(w, one_hot_dict['<UNK>']) for w in sentence.split()])

    length = [len(sentence.split())]
    encoder_feed_dict = {sentence_placeholder: sentence_matrix, sent_lengths_placeholder: length}
    sentence_encoder_outputs = sess.run(encoder_output_tensors, feed_dict= encoder_feed_dict)
    decoder_feed_dict = {encoder_output_tensors[i] : sentence_encoder_outputs[i]
                         for i in range(len(encoder_output_tensors))}

    if not learn_embeddings:
        W_we = tf.get_default_graph().get_tensor_by_name('W_we:0')
        encoder_feed_dict = union_dicts(encoder_feed_dict, {W_we : embeddings_matrix})
    return encoder_feed_dict, decoder_feed_dict


def get_feed_dicts_from_program(program, logical_tokens_embeddings_dict, program_dependent_placeholders,
                                skipped_indices = []):
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


# build the computaional graph:
g_1 = tf.Graph()
#with g_1.as_default():
# bi-lstm encoder - given a sentence (of a variable length) as a sequence of word embeddings,
# and returns the lstm outputs.

sentence_placeholder, sent_lengths_placeholder, h, e_m = build_sentence_encoder(vocabulary_size=len(one_hot_dict))

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

logits = tf.transpose(token_unnormalized_dist) # + invalid_logical_tokens_mask

# cross-entropy loss per single token in a single sentence
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    labels=chosen_logical_tokens, logits=logits))


def run_model(sess, dataset, mode, validation_dataset = None, load_params_path = None, save_model_path = None,
              return_sentences_results=None, beam_classifier=False, beam_classifier_test=False, clf_params_path=None):
    """ 
    a method for running the weakly-supervised model
    
    :param sess: a tf Session in the context of which the model is to be run
    :param dataset: a CNLVRDataSet object on which to run, whether in train or in teat mode.
    :param mode: either 'train' or 'test'
    :param validation_dataset: another CNLVRDataSet object, that's used for validation when running in train mode.
    :param load_params_path: a path to pre-trained weights
    :param save_model_path: a path for saving the learned model
    
    
    note: this method is used for either training or testing the weakly-supervised model,
    according to the 'mode' argument.
    """

    modes = ('train', 'test')
    assert mode in modes
    test_between_training_epochs = validation_dataset is not None
    if (mode=='test'):
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
    gradList = sess.run(theta) # just to get dimensions right
    gradBuffer = {}
    for i, grad in enumerate(gradList):
        gradBuffer[i] = grad*0

    n_epochs = 1 if mode=='test' else MAX_N_EPOCHS
    stopping_criterion_met = False

    # initialize variables for collecting statistics during the run
    num_consistent_per_sentence, beam_final_sizes, beam_final_sizes_before, mean_program_lengths = [], [], [], []
    accuracy_by_prog_rank , num_consistent_by_prog_rank, incorrect_parses, all_consistent_decodings = {}, {} , {}, {}
    n_consistent_top_by_model = []
    n_consistent_top_by_reranking = []
    n_ccorrect_top_by_model = []
    n_correct_top_by_reranking= []
    n_consistent_top_by_classifier = []
    n_correct_top_by_classifier = []
    timer = []
    epochs_completed, num_have_pattern, iter, n_samples = 0, 0, 0, 0
    batch_num_in_epoch = 1
    classifier_ran_already = False

    # a dictionary for cached programs
    if USE_CACHED_PROGRAMS and LOAD_CACHED_PROGRAMS and mode=='train':
        cpf = open(CACHED_PROGRAMS, 'rb')
        all_cached_programs = pickle.load(cpf)
    else:
        all_cached_programs = {}


    start = time.time()

    curr_dataset, other_dataset = dataset, validation_dataset
    curr_mode = mode
    other_mode = 'train' if mode=='test' else 'test'

    stats_for_all_sentences = {}

    all_features = []
    all_labels = []

    while dataset.epochs_completed < n_epochs and not stopping_criterion_met:

        # get a mini-batch of sentences and their related images
        batch, is_last_batch_in_epoch = curr_dataset.next_batch(BATCH_SIZE_UNSUPERVISED)
        batch_sentence_ids = [key for key in batch.keys()]
        sentences, samples = zip(*[batch[k] for k in batch_sentence_ids])

        # get a dictionary of the current embeddings of the logical tokens
        # (in train they are change after every mini batch - as they are also trainable variables)
        current_logical_tokens_embeddings = sess.run(W_logical_tokens)
        logical_tokens_embeddings_dict = \
            {token : current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens_ids}

        for step in range (len(sentences)):
            # Go over the data set sentence by sentence (instead of sample by sample),
            # for efficiency and in order to get a more reliable signal for the reward (each parsing
            # will be checked against all four images related to the sentence).

            sentence_id = batch_sentence_ids[step]
            sentence = (sentences[step])
            related_samples = samples[step]

            decodings_from_cache = []
            if USE_CACHED_PROGRAMS and curr_mode == 'train':
                # find the N top-rated candidate programs for parsing the given sentence.
                # the rating is based on previous rewards of these programs on sentences that
                # are similar in pattern to the current sentence.
                decodings_from_cache = get_programs_for_sentence_by_pattern(sentence, all_cached_programs)
                decodings_from_cache = [token_seq.split() for token_seq in decodings_from_cache][:N_CACHED_PROGRAMS]
                num_have_pattern += 1 if decodings_from_cache else 0

            # create feed dicts for running the current sentence in the nn.
            # running the sentence through the encoder lstm to obtain its output is also done inside the following call.
            # This time-consuming operation is executed only once, and doesn't need to be
            # repeated at each step during the beam search.
            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, (h, e_m), LEARN_EMBEDDINGS)

            # define a method for getting token probabilities, given a partial program and the
            # current state of the nn.
            next_token_probs_getter = \
                lambda partial_prog :  get_next_token_probs_from_nn(partial_prog,
                                                                    logical_tokens_embeddings_dict,
                                                                    decoder_feed_dict,
                                                                    history_embedding_placeholder,
                                                                    token_prob_dist_tensor,
                                                                    )

            epsilon_for_beam = 0.0 if curr_mode == 'test' else EPSILON_FOR_BEAM_SEARCH

            original_sentence_for_beam = sentence if SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH else None

            # perform beam search to obtain candidate parsings for the sentence
            beam = e_greedy_randomized_beam_search(next_token_probs_getter,
                                                   logical_tokens_mapping,
                                                   original_sentence= original_sentence_for_beam,
                                                   epsilon=epsilon_for_beam,
                                                   suggested_decodings=decodings_from_cache)

            programs_execution_results = {}
            programs_from_cache = []

            # run all programs in beam on all images related to the sentence and get the results
            for prog in beam:
                if prog.token_seq in decodings_from_cache:
                    programs_from_cache.append(prog)

                programs_execution_results[prog] = get_program_execution_stats(
                    prog.token_seq, related_samples, logical_tokens_mapping)

            # update the dictionary of cached programs by sentence pattern according to
            # the execution results of the programs in the beam

            if USE_CACHED_PROGRAMS and curr_mode =='train':
                for prog, prog_stats in programs_execution_results.items():
                    if prog_stats.is_consistent or prog in programs_from_cache:
                        update_programs_cache(all_cached_programs, sentence, prog, prog_stats)


            # consistent programs are those that compile and correctly predicts the labels of each of the images
            consistent_programs = [prog for prog, stats in programs_execution_results.items() if stats.is_consistent]

            if curr_mode =='train':
                # in order to mitigate the problem of spurious programs, a program for a given sentence gets
                #  a reward iff it is consistent with all its related images, (i.e. compiles and returns the correct
                # label on all samples related to the sentence).
                # here we take the gradient w.r.t each such consistent program, and weight those gradients
                # using the beta-meritocratic approach presented in Goo at el. 2017
                # we add the gradients to a buffer and take an optimizer step after each batch

                programs_gradient_weights = get_gradient_weights_for_programs(consistent_programs)

                for prog_idx, program in enumerate(consistent_programs):
                    program_dependent_feed_dict = get_feed_dicts_from_program(
                        program, logical_tokens_embeddings_dict, program_dependent_placeholders)
                    program_grad = sess.run(
                        compute_program_grads, feed_dict=union_dicts(encoder_feed_dict, program_dependent_feed_dict))

                    for i, (grad, var) in enumerate(program_grad):
                        if (np.isnan(grad)).any():
                            raise RuntimeError("NaN gradients encountered in {}".format(i))
                        gradBuffer[i] += programs_gradient_weights[prog_idx] * grad

            # get statistics for this iteration

            beam_final_sizes.append(len(beam))
            num_consistent_per_sentence.append(len(consistent_programs))

            if beam:
                # gather some statistics about the programs in the beam
                mean_program_lengths.append(np.mean([len(pp) for pp in beam]))
                beam_reranked = beam_reranker(sentence, beam, words_to_tokens)

                top_program_by_model = beam[0]
                top_program_by_reranking = beam_reranked[0]

                top_by_model_stats = programs_execution_results[top_program_by_model]
                top_by_reranking_stats = programs_execution_results[top_program_by_reranking]
                top_by_classifier_stats = None

                if beam_classifier:
                    features = []
                    labels = []
                    for prog in beam:
                        feat_vector = get_features(sentence, prog)
                        label = programs_execution_results[prog].is_consistent
                        features.append(feat_vector)
                        labels.append(label)
                    all_features.append(features)
                    all_labels.append(labels)

                    if beam_classifier_test:
                        with tf.Session(graph=g_2) as sess2:

                            classifier_best_index = run_beam_classification(sess2,[features],[labels],BEAM_SIZE,inference=True,load_params_path=clf_params_path,reuse=classifier_ran_already)
                            classifier_ran_already = True
                            if classifier_best_index >= len(beam):
                                top_program_by_classifier = beam[0]
                            else:
                                top_program_by_classifier = beam[classifier_best_index]
                            top_by_classifier_stats = programs_execution_results[top_program_by_classifier]


            if not beam: # no valid program was found by the beam - default is to guess 'True' for all images
                top_by_model_stats = top_by_reranking_stats = get_program_execution_stats(
                    ["ERR"], related_samples, logical_tokens_mapping)

            if mode == 'test':
                stats_for_all_sentences[sentence_id] = {'top_program_by_reranking' : top_program_by_reranking,
                                                        'top_by_reranking_stats' : top_by_reranking_stats,
                                                        'top_program_by_model' : top_program_by_model,
                                                        'top_by_model_stats' : top_by_model_stats,
                                                        'top_program_by_classifier': top_program_by_classifier,
                                                        'top_by_classifier_stats': top_by_classifier_stats,
                                                        'consistent_programs' : consistent_programs,
                                                        'beam_reranked' : beam_reranked,
                                                        'samples' : samples}

            n_consistent_top_by_model.append(top_by_model_stats.is_consistent)
            n_consistent_top_by_reranking.append(top_by_reranking_stats.is_consistent)
            n_consistent_top_by_classifier.append(top_by_classifier_stats.is_consistent)
            n_ccorrect_top_by_model.append(top_by_model_stats.n_correct)
            n_correct_top_by_reranking.append(top_by_reranking_stats.n_correct)
            n_correct_top_by_classifier.append(top_by_classifier_stats.n_correct)



            n_samples += len(related_samples)
            iter += 1

        stop = time.time()
        timer.append(stop-start)
        start = time.time()
        print(".")

        if curr_mode == 'train':
            for i, g in enumerate(gradBuffer):
                gradBuffer[i] = gradBuffer[i]/len(sentences)

            sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})
            for i, grad in enumerate(gradBuffer):
                gradBuffer[i] = gradBuffer[i]*0


        if batch_num_in_epoch % PRINT_EVERY == 0 or is_last_batch_in_epoch:
            # print the result every PRINT_EVERY minibatches, and in addition print
            # to a .txt file after every epoch

            stats_file = open(STATS_FILE, 'a')
            if is_last_batch_in_epoch:

                print_to = (orig_stdout,stats_file )
            else:
                print_to =  [orig_stdout]


            for print_target in print_to:
                sys.stdout = print_target
                print("##############################")
                print("current mode = " + curr_mode)
                print("current dataset = " + curr_dataset.name)


                if is_last_batch_in_epoch:
                    print(time.strftime("%Y-%m-%d %H:%M"))
                    print("finished epoch {}.".format(curr_dataset.epochs_completed))
                    if PRINT_PARAMS:
                        print("parameters used in learning:")
                        for param_name in ['EPSILON_FOR_BEAM_SEARCH', 'BETA', 'SKIP_AUTO_TOKENS', 'N_CACHED_PROGRAMS',
                                           'INJECT_TO_BEAM',
                                            'SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH',
                                           'AVOID_ALL_TRUE_SENTENCES']:
                            print ("{0} : {1}".format(param_name, eval(param_name)))
                    print("stats for this epoch:")


                else:
                    print("epoch {}".format(curr_dataset.epochs_completed + 1))
                    print("finished {0} mini batches within {1:.2f} seconds".format(batch_num_in_epoch, np.sum(timer)))
                    print("number of sentences in data = {}".format(curr_dataset.num_examples))
                    print("stats for this epoch so far:")
                mean_consistent = np.mean(num_consistent_per_sentence)
                mean_beam_size = np.mean(beam_final_sizes)
                n_non_zero = np.count_nonzero(num_consistent_per_sentence)

                print("{0} out of {1} sentences had consistent programs in beam".format(n_non_zero, len(
                    num_consistent_per_sentence)))
                print("mean consistent parses for sentence = {0:.2f}, mean beam size for sentence = {1:.2f}" \
                      ",  mean prog length = {2:.2f}".format(
                    mean_consistent, mean_beam_size, np.mean(mean_program_lengths)))

                print('top programs by model had so far {0} correct answers out of {1} samples ({2:.2f}%), and '
                    '{3} consistent parses out of {4} sentences ({5:.2f}%)'.format(sum(n_ccorrect_top_by_model),
                                                                        n_samples,
                                                                        (sum(n_ccorrect_top_by_model) / n_samples) * 100,
                                                                        sum(n_consistent_top_by_model),
                                                                        iter,
                                                                        (sum(n_consistent_top_by_model) / iter) * 100))
                print('top programs by reranking had so far {0} correct answers out of {1} samples ({2:.2f}%), and '
                    '{3} consistent parses out of {4} sentences ({5:.2f}%)'.format(sum(n_correct_top_by_reranking),
                                                                        n_samples,
                                                                        (sum(n_correct_top_by_reranking) / n_samples) * 100,
                                                                        sum(n_consistent_top_by_reranking),
                                                                        iter,
                                                                        (sum(n_consistent_top_by_reranking) / iter) * 100))
                print('top programs by classifier had so far {0} correct answers out of {1} samples ({2:.2f}%), and '
                      '{3} consistent parses out of {4} sentences ({5:.2f}%)'.format(sum(n_correct_top_by_classifier),
                                                                                     n_samples,
                                                                                     (sum(
                                                                                         n_correct_top_by_classifier) / n_samples) * 100,
                                                                                     sum(n_consistent_top_by_classifier),
                                                                                     iter,
                                                                                     (sum(
                                                                                         n_consistent_top_by_classifier) / iter) * 100))
                print("##############################")

            sys.stdout = orig_stdout
            stats_file.close()
        if is_last_batch_in_epoch:
            num_consistent_per_sentence , beam_final_sizes, n_ccorrect_top_by_model, n_correct_top_by_reranking\
                = [], [] , [], []
            n_consistent_top_by_reranking, n_consistent_top_by_model, timer = [], [], []
            n_samples,iter = 0, 0

            batch_num_in_epoch=0

            if curr_mode == 'train':
                if save_model_path :
                    saver2.save(sess, save_model_path, global_step=dataset.epochs_completed,write_meta_graph=False)
                    print("saved epoch %d" % dataset.epochs_completed)

                if SAVE_CACHED_PROGRAMS:
                    cpf = open(CACHED_PROGRAMS, 'wb')
                    pickle.dump(all_cached_programs, cpf)

        batch_num_in_epoch += 1

        if is_last_batch_in_epoch and test_between_training_epochs:
            curr_mode, other_mode = other_mode, curr_mode
            curr_dataset, other_dataset = other_dataset, curr_dataset

    if mode == 'test' and return_sentences_results:
        # f = save_stats_path + '_' + curr_dataset.name + '_' + time.strftime("%Y-%m-%d_%H_%M")
        #pickle.dump(stats_for_all_sentences, open(f, 'wb'))
        return stats_for_all_sentences

    if beam_classifier and not beam_classifier_test :
        return all_features, all_labels

    return {}


def run_supervised_training(sess, load_params_path = None, save_params_path = None, num_epochs = MAX_N_EPOCHS):
    """
    a method for training the supervised model using a dataset of sentence-logical form pairs
    :param sess: a tf.Session in the context of which to run
    :param load_params_path: for loading pre-trained weights
    :param save_params_path: for saving the weights learned
    
    
    this methods runs through the data-set and optimzes using gradient descent on the cross-entropy
    of the model output with the real tokens from the given logical forms.
    """

    theta = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    compute_program_grads = optimizer.compute_gradients(cross_entropy)
    batch_grad = build_batchGrad()
    update_grads = optimizer.apply_gradients(zip(batch_grad, theta))

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(theta)

    if load_params_path:
        saver.restore(sess,load_params_path)

    # load data
    train_set = DataSetForSupervised(definitions.SUPERVISED_TRAIN_PICKLE)
    validation_set = DataSetForSupervised(definitions.SUPERVISED_VALIDATION_PICKLE)

    #initialize gradients
    gradList = sess.run(theta) # just to get dimensions
    gradBuffer = {}
    for var, grad in enumerate(gradList):
        gradBuffer[var] = grad*0

    accuracy, accuracy_chosen_tokens, epoch_losses, epoch_log_prob = [] , [], [], []
    batch_num , epoch_num = 0, 0
    current_data_set = train_set
    statistics = {train_set : [], validation_set : []}

    while epoch_num < num_epochs:
        if current_data_set.epochs_completed != epoch_num:
            statistics[current_data_set].append((np.mean(epoch_losses), np.mean(accuracy), np.mean(accuracy_chosen_tokens)))
            print("epoch number {0}: mean loss = {1:.3f}, mean accuracy = {2:.3f}, mean accuracy ignore automatic = {3:.3f},"
                  "epoch mean log probability = {4:.4f}".
                    format(epoch_num, np.mean(epoch_losses), np.mean(accuracy), np.mean(accuracy_chosen_tokens),
                           np.mean(epoch_log_prob)))
            accuracy, accuracy_chosen_tokens, epoch_losses, epoch_log_prob = [], [], [], []
            if current_data_set == train_set:
                current_data_set = validation_set
            else:
                current_data_set = train_set
                epoch_num+= 1

        sentences, labels = zip(*current_data_set.next_batch(BATCH_SIZE_SUPERVISED))
        batch_num += 1
        batch_size = len(sentences)

        current_logical_tokens_embeddings = sess.run(W_logical_tokens)
        logical_tokens_embeddings_dict = \
            {token : current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens_ids}

        for step in range(batch_size):
            sentence = sentences[step]
            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, (h, e_m), LEARN_EMBEDDINGS_IN_PRETRAIN)

            next_token_probs_getter = lambda pp :  get_next_token_probs_from_nn(pp, logical_tokens_embeddings_dict,
                                                                                decoder_feed_dict,
                                                                                history_embedding_placeholder,
                                                                                token_prob_dist_tensor)

            golden_parsing = labels[step].split()
            try:
                golden_program, (valid_tokens_history, greedy_choices) = \
                    program_from_token_sequence(next_token_probs_getter, golden_parsing, logical_tokens_mapping)
            except(ValueError) as ve:
                continue

            epoch_log_prob.append(golden_program.logprob)
            no_real_choice_indices = [i for i in range(len(golden_parsing))if len(valid_tokens_history[i])==1]
            skipped = [] if AUTOMATIC_TOKENS_IN_GRAD else no_real_choice_indices

            program_dependent_feed_dict = get_feed_dicts_from_program(golden_program,
                                                                      logical_tokens_embeddings_dict,
                                                                      program_dependent_placeholders,
                                                                      skipped_indices= skipped)

            correct_greedy_choices = [golden_parsing[i] == greedy_choices[i] for i in range(len(golden_parsing))]
            accuracy.append(sum(correct_greedy_choices)/ len(golden_parsing))
            accuracy_chosen_tokens.append(sum(correct_greedy_choices[i] for i in range(len(golden_parsing))
                                              if i not in no_real_choice_indices)/ (len(golden_parsing)- len(no_real_choice_indices)))


            # calculate gradient
            program_grad, loss = sess.run([compute_program_grads, cross_entropy],
                                        feed_dict= union_dicts(encoder_feed_dict, program_dependent_feed_dict))
            epoch_losses.append(loss)


            for var, grad in enumerate(program_grad):
                gradBuffer[var] += grad[0]
        if current_data_set is train_set:
            sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})


        for var, grad in enumerate(gradBuffer):
            gradBuffer[var] = gradBuffer[var]*0


    if save_params_path:
        saver.save(sess, save_params_path)



if __name__ == '__main__':

    orig_stdout = sys.stdout
    #with tf.Session() as sess:
    #    run_supervised_training(sess,save_params_path=PRE_TRAINED_WEIGHTS)
    weights_from_supervised_pre_training = PRE_TRAINED_WEIGHTS
    best_weights_so_far = TRAINED_WEIGHTS_BEST
    time_stamp = time.strftime("%Y-%m-%d_%H_%M")
    OUTPUT_WEIGHTS = os.path.join(SEQ2SEQ_DIR, 'learnedWeightsWeaklySupervised', 'weights_' + time_stamp + '.ckpt')
    STATS_FILE = os.path.join(SEQ2SEQ_DIR, 'running logs', 'stats_' + time_stamp + '.txt')
    SENTENCES_RESULTS_FILE_DEV = os.path.join(SEQ2SEQ_DIR,
                                              'running logs', 'sentences_results_dev_' +
                                              time_stamp + '.txt')
    SENTENCES_RESULTS_FILE_TEST = os.path.join(SEQ2SEQ_DIR,
                                              'running logs', 'sentences_results_test_' +
                                              time_stamp + '.txt')
    SENTENCES_RESULTS_FILE_TEST2 = os.path.join(SEQ2SEQ_DIR,
                                              'running logs', 'sentences_results_test2_' +
                                              time_stamp + '.txt')


    train_dataset = CNLVRDataSet(DataSet.TRAIN)
    dev_dataset = CNLVRDataSet(DataSet.DEV)
    test_dataset = CNLVRDataSet(DataSet.TEST)
    #test2_dataset = CNLVRDataSet(DataSet.TEST2)

    run_train = False # change to True if you really want to run the whole thing...

    if run_train:
        # training the weakly supervised model with weights initialized to the values learned in the supervises learning.
        # this takes about 6 hours running on a CPU. The specific hyper-parameters used here, as well as
        # whether other techniques/variants described in the paper are used (e.d. sentence-driven constrains on the beam,
        # beam re-reanking, auto-completion of tokens etc.) are all set in hyper_params.py.
        # the results are printed to STATS_FILE after every epoch


        with tf.Session() as sess:
            run_model(sess, train_dataset, mode='train', validation_dataset=dev_dataset,
                      load_params_path=weights_from_supervised_pre_training, save_model_path=OUTPUT_WEIGHTS)

    # running a test on the dev and test datasets, using the weights that achieved the best accuracy and consistency
    # rates that were presented in our paper. The accuracy results are printed and saved to to STATS_FILE,
    # and the results by sentence are saved to  SENTENCES_RESULTS_FILE_DEV and SENTENCES_RESULTS_FILE_TEST.

    run_beam_reranking = True
    beam_reranking_train = False
    if run_beam_reranking:
        beam_classifier_weights_path = os.path.join(SEQ2SEQ_DIR, 'beamClassificationWeights' + time_stamp + '.ckpt')
        #tf.reset_default_graph() # set as comment for inference
        with tf.Session() as sess:
            if beam_reranking_train:
                #features, labels = run_model(sess, train_dataset, mode='test',
                            #load_params_path=best_weights_so_far, beam_classifier=True)
                #pickle.dump(features,open(os.path.join(SEQ2SEQ_DIR,'features_for_beam_reranking'), 'wb'))
                #pickle.dump(labels, open(os.path.join(SEQ2SEQ_DIR,'labels_for_beam_reranking'), 'wb'))
                features = pickle.load(open(os.path.join(SEQ2SEQ_DIR,'features_for_beam_reranking'), 'rb'))
                labels = pickle.load(open(os.path.join(SEQ2SEQ_DIR,'labels_for_beam_reranking'), 'rb'))
                run_beam_classification(sess,features,labels,BEAM_SIZE,save_path=beam_classifier_weights_path)
            else:
                run_model(sess, test_dataset, mode='test',
                          load_params_path=best_weights_so_far, return_sentences_results=True,beam_classifier=True,
                          beam_classifier_test=True, clf_params_path=os.path.join(SEQ2SEQ_DIR,'beamClassificationWeights2017-10-30_21_21.ckpt'))
        exit(0)

    dev_results_by_sentence , test_results_by_sentence, test2_results_by_sentence = {},{}, {}

    dev_dataset.restart()
    with tf.Session() as sess:
        dev_results_by_sentence = run_model(sess, dev_dataset, mode='test',
                                            load_params_path=best_weights_so_far, return_sentences_results=True)

    save_sentences_test_results(dev_results_by_sentence, dev_dataset, SENTENCES_RESULTS_FILE_DEV)

    test_dataset.restart()
    with tf.Session() as sess:
        test_results_by_sentence = run_model(sess, test_dataset, mode='test',
                                            load_params_path=best_weights_so_far, return_sentences_results=True)

    save_sentences_test_results(test_results_by_sentence, test_dataset, SENTENCES_RESULTS_FILE_TEST)

    #test2_dataset.restart()
    #with tf.Session() as sess:
    #    test2_results_by_sentence = run_model(sess, test2_dataset, mode='test',
    #                                        load_params_path=best_weights_so_far, return_sentences_results=True)

    #save_sentences_test_results(test2_results_by_sentence, test2_dataset, SENTENCES_RESULTS_FILE_TEST2)

