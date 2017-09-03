import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import pickle
import numpy as np
import os
import definitions
from seq2seqModel.utils import *
from seq2seqModel.logical_forms_generation import *
from handle_data import CNLVRDataSet, SupervisedParsing
from seq2seqModel.beam import *
from general_utils import increment_count, union_dicts
import definitions
import time

#paths

LOGICAL_TOKENS_MAPPING_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping_limitations')
WORD_EMBEDDINGS_PATH = os.path.join(definitions.ROOT_DIR, 'word2vec', 'embeddings_10iters_12dim')
PARSED_EXAMPLES_T = os.path.join(definitions.DATA_DIR, 'parsed sentences', 'parses for check as tokens')
TRAINED_WEIGHTS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables.ckpt')
TRAINED_WEIGHTS2 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables2.ckpt')
TRAINED_WEIGHTS3 = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables3.ckpt')
TRAINED_UNS_WEIGHTS = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeightsUns','trained_variables3.ckpt')
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
USE_CACHED_PROGRAMS = False


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
    lstm_fw_cell = BasicLSTMCell (LSTM_HIDDEN_SIZE, forget_bias=1.0)
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


def build_batchGrad():
    lstm_fw_weights_grad = tf.placeholder(tf.float32, name="lstm_fw_weights_grad")
    lstm_fw_bias_grad = tf.placeholder(tf.float32, name="lstm_fw_bias_grad")
    lstm_bw_weights_grad = tf.placeholder(tf.float32, name="lstm_bw_weights_grad")
    lstm_bw_bias_grad = tf.placeholder(tf.float32, name="lstm_bw_bias_grad")
    wq_grad = tf.placeholder(tf.float32, name="wq_grad")
    wa_grad = tf.placeholder(tf.float32, name="wa_grad")
    ws_grad = tf.placeholder(tf.float32, name="ws_grad")
    logical_tokens_grad = tf.placeholder(tf.float32, name="logical_tokens_grad")
    batchGrad = [lstm_fw_weights_grad, lstm_fw_bias_grad, lstm_bw_weights_grad, lstm_bw_bias_grad,
                 wq_grad, wa_grad, ws_grad, logical_tokens_grad]
    return batchGrad

def get_next_token_probs(partial_program, logical_tokens_embeddings_dict, decoder_feed_dict, history_embedding_tensor,
                         token_prob_dist):
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


    valid_next_tokens = partial_program.get_possible_continuations()
    probs_given_valid = [1.0] if len(valid_next_tokens) == 1 else \
        [current_probs[logical_tokens_ids[next_tok]] for next_tok in valid_next_tokens]
    probs_given_valid = probs_given_valid / np.sum(probs_given_valid)
    return valid_next_tokens, probs_given_valid


def get_gradient_weights_for_beam(beam_rewarded_programs):

    beam_log_probs = np.array([prog.logprob for prog in beam_rewarded_programs])
    q_mml = softmax(beam_log_probs)
    return np.power(q_mml, BETA) / np.sum(np.power(q_mml, BETA))


def sentences_to_embeddings(sentences, embeddings_dict):
    return np.array([[embeddings_dict.get(w, embeddings_dict['<UNK>']) for w in sentence] for sentence in sentences])


def get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, encoder_output_tensors):

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







def run_unsupervised_training(sess, load_params_path = None, save_model_path = None):
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

    logits = tf.transpose(token_unnormalized_dist) # + invalid_logical_tokens_mask

    # cross-entropy loss per single token in a single sentence
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=chosen_logical_tokens, logits=logits))


    # the log-probability according to the model of a program given an input sentnce.
    #program_log_prob = tf.reduce_sum(tf.log(token_prob_dist_tensor) * tf.transpose(chosen_logical_tokens))
    theta = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    compute_program_grads = optimizer.compute_gradients(cross_entropy)
    batch_grad = build_batchGrad()
    update_grads = optimizer.apply_gradients(zip(batch_grad, theta))

    init = tf.global_variables_initializer()
    sess.run(init)

    if load_params_path:
        tf.train.Saver(theta).restore(sess, load_params_path)

    gradList = sess.run(theta) # just to get dimensions right
    gradBuffer = {}

    # load data
    train = CNLVRDataSet(definitions.TRAIN_JSON, ignore_all_true = False)
    #train.sort_sentences_by_complexity(4)
    #train.choose_levels_for_curriculum_learning([0,1,2,3])
    file = open(SENTENCES_IN_PRETRAIN_PATTERNS, 'rb')
    sentences_in_pattern = pickle.load(file)
    file.close()
    train.use_subset_by_sentnce_condition(lambda s: s in sentences_in_pattern.values())

    #initialize gradients
    for var, grad in enumerate(gradList):
        gradBuffer[var] = grad*0
    batch_num = 1
    total_correct = 0
    correct_beam_parses = open("correct beams 9.txt ", 'w')
    num_consistent_per_sentence, beam_final_sizes, mean_program_lengths = [], [], []
    accuracy_by_prog_rank , num_consistent_by_prog_rank = {} , {}
    incorrect_parses = {}
    n_images = 0
    iter = 0
    start = time.time()
    while train.epochs_completed < 1:

        batch = train.next_batch(BATCH_SIZE_UNSUPERVISED)
        ids = [key for key in batch.keys()]

        sentences, samples = zip(*[batch[k] for k in ids])
        current_logical_tokens_embeddings = sess.run(W_logical_tokens)
        logical_tokens_embeddings_dict = \
            {token : current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens_ids}

        for step in range (len(sentences)):
            iter += 1
            sentence = (sentences[step])
            sentence_id = ids[step]
            related_samples = samples[step]
            n_images+= len(related_samples)

            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, (h, e_m))

            next_token_probs_getter = lambda pp :  get_next_token_probs(pp, logical_tokens_embeddings_dict,
                                                                        decoder_feed_dict,
                                                                        history_embedding_placeholder,
                                                                        token_prob_dist_tensor)

            beam = e_greedy_randomized_beam_search_udi(next_token_probs_getter, logical_tokens_mapping,
                                                   original_sentence= sentence)

            rewarded_programs = []

            compiled = 0
            correct = 0
            actual_labels = np.array([sample.label for sample in related_samples])
            print ("{0} : {1}".format(sentence_id, sentence))
            for prog_rank, prog in enumerate(beam):
                # execute program and get reward is result is same as the label

                execution_results = np.array([execute(prog.token_seq,sample.structured_rep,logical_tokens_mapping)
                                     for sample in related_samples])

                compiled+= sum(res is not None for res in execution_results)
                for i in range(len(execution_results)):
                    if execution_results[i] is None:
                        execution_results[i] = True
                reward = 1 if all(execution_results==actual_labels) else 0
                if prog_rank<5:
                    #print(prog_rank, prog, 'correct' if reward else 'incorrect')
                    if prog_rank==0 and not reward:
                        incorrect_parses[sentence_id] = (sentence, prog)

                increment_count(accuracy_by_prog_rank, prog_rank, sum(execution_results==actual_labels))

                correct+=reward

                if reward>0:
                    increment_count(num_consistent_by_prog_rank, prog_rank)
                    correct_beam_parses.write(sentence +"\n")
                    correct_beam_parses.write(" ".join(prog.token_seq)+"\n")
                    rewarded_programs.append(prog)

            #print("beam size = {0}, {1} programs compiled, {2} correct".format(len(beam), compiled, correct))

            #print("beam, compliled, correct = {0} /{1} /{2}".format(len(beam),compiled ,correct))

            if beam:
                mean_program_lengths.append(np.mean([len(pp) for pp in beam]))

            else:
                # if bean is empty predict True for each image
                for k in accuracy_by_prog_rank.keys():
                    increment_count(accuracy_by_prog_rank, k, np.count_nonzero(actual_labels))


            num_consistent_per_sentence.append(correct)
            beam_final_sizes.append(len(beam))

            if not rewarded_programs:
                continue

            programs_gradient_weights = get_gradient_weights_for_beam(rewarded_programs)

            for idx, program in enumerate(rewarded_programs):
                program_dependent_feed_dict = \
                    get_feed_dicts_from_program(program, logical_tokens_embeddings_dict, program_dependent_placeholders)
                program_grad = sess.run(compute_program_grads, feed_dict =
                                        union_dicts(encoder_feed_dict, program_dependent_feed_dict))

                for var,grad in enumerate(program_grad):
                    if (np.isnan(grad)).any():
                        print("nan gradient")
                    gradBuffer[var] +=  programs_gradient_weights[idx] * grad[0]

        stop = time.time()
        print("time for mini batch = {0:.2f} seconds".format(stop - start))
        start = time.time()
        if batch_num % 10 == 0:
            #print("accuracy: %.2f" % (total_correct / (batch_size_unsupervised * 10)))
            mean_corect = np.mean(num_consistent_per_sentence)
            mean_beam_size = np.mean(beam_final_sizes)
            n_non_zero = np.count_nonzero(num_consistent_per_sentence)
            print("{0} out of {1} sentences had consistent programs in beam".format(n_non_zero, len(num_consistent_per_sentence)))
            print("mean consistent parses for sentence = {0:.2f}, mean beam size for sentence = {1:.2f}"\
                  ", mean prog length = {2:.2f}".format(
                mean_corect, mean_beam_size, np.mean(mean_program_lengths)))

            num_consistent_per_sentence, beam_final_sizes, mean_program_lengths = [], [], []

            for k,v in sorted(accuracy_by_prog_rank.items(), key= lambda kvp : kvp[1], reverse=True)[:1]:
                print ('programs ranked {0} in beam had so far {1} correct answers out of {2} samples'.format(k,v, n_images))
            for k,v in sorted(num_consistent_by_prog_rank.items(), key= lambda kvp : kvp[1], reverse=True)[:1]:
                print ('programs ranked {0} in beam had so far {1} consistent parses out of {2} sentences'.format(k,v, iter))


        batch_num += 1
        for i, g in enumerate(gradBuffer):
            gradBuffer[i] = gradBuffer[i]/len(sentences)

        sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})
        for var, grad in enumerate(gradBuffer):
            gradBuffer[var] = gradBuffer[var]*0

    if save_model_path:
        tf.train.Saver().save(sess, save_model_path)
    return


def run_supervised_training(sess, load_params_path = None, save_params_path = None):
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




    logits = tf.transpose(token_unnormalized_dist) # + invalid_logical_tokens_mask

    # cross-entropy loss per single token in a single sentence
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=chosen_logical_tokens, logits=logits))

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
    train = SupervisedParsing(definitions.SUPERVISED_TRAIN_PICKLE)
    validation = SupervisedParsing(definitions.SUPERVISED_VALIDATION_PICKLE)

    #initialize gradients
    gradList = sess.run(theta) # just to get dimensions
    gradBuffer = {}
    for var, grad in enumerate(gradList):
        gradBuffer[var] = grad*0

    accuracy, accuracy_chosen_tokens, epoch_losses, epoch_log_prob = [] , [], [], []
    batch_num , epoch_num = 0, 0
    current_data_set = train
    statistics = {train : [], validation : []}

    while epoch_num < 8:
        if current_data_set.epochs_completed != epoch_num:
            statistics[current_data_set].append((np.mean(epoch_losses), np.mean(accuracy), np.mean(accuracy_chosen_tokens)))
            print("epoch number {0}: mean loss = {1:.3f}, mean accuracy = {2:.3f}, mean accuracy ignore automatic = {3:.3f},"
                  "epoch mean log probability = {4:.4f}".
                    format(epoch_num, np.mean(epoch_losses), np.mean(accuracy), np.mean(accuracy_chosen_tokens),
                           np.mean(epoch_log_prob)))
            accuracy, accuracy_chosen_tokens, epoch_losses, epoch_log_prob = [], [], [], []
            if current_data_set == train:
                current_data_set = validation
            else:
                current_data_set = train
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
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, (h, e_m))

            next_token_probs_getter = lambda pp :  get_next_token_probs(pp, logical_tokens_embeddings_dict,
                                                                        decoder_feed_dict,
                                                                        history_embedding_placeholder,
                                                                        token_prob_dist_tensor)

            golden_parsing = labels[step].split()
            golden_program, (valid_tokens_history, greedy_choices) = \
                program_from_token_sequence(next_token_probs_getter, golden_parsing, logical_tokens_mapping)
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
        if current_data_set is train:
            sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})


        for var, grad in enumerate(gradBuffer):
            gradBuffer[var] = gradBuffer[var]*0


    if save_params_path:
        saver.save(sess, save_params_path)

def run_inference(sess, data, load_params):
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

    logits = tf.transpose(token_unnormalized_dist) # + invalid_logical_tokens_mask


    # the log-probability according to the model of a program given an input sentnce.
    #program_log_prob = tf.reduce_sum(tf.log(token_prob_dist_tensor) * tf.transpose(chosen_logical_tokens))
    theta = tf.trainable_variables()
    init = tf.global_variables_initializer()
    sess.run(init)

    if load_params:
        tf.train.Saver(theta).restore(sess, load_params)

    total = 0
    samples_num = len(data.samples.values())
    correct_avg, correct_first, empty_beam = 0, 0, 0

    current_logical_tokens_embeddings = sess.run(W_logical_tokens)
    logical_tokens_embeddings_dict = \
        {token: current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens_ids}

    for sample in data.samples.values():
        if total % 10 == 0:
            print("sample %d out of %d" % (total,samples_num))

        sentences = [sample.sentence]
        label = sample.label

        for step in range (len(sentences)):
            sentence = (sentences[step])

            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, (h, e_m))

            next_token_probs_getter = lambda pp :  get_next_token_probs(pp, logical_tokens_embeddings_dict,
                                                                        decoder_feed_dict,
                                                                        history_embedding_placeholder,
                                                                        token_prob_dist_tensor)

            beam = e_greedy_randomized_beam_search(next_token_probs_getter, logical_tokens_mapping,
                                                   original_sentence= sentence)

            execution_results = []

            for prog_rank, prog in enumerate(beam):
                # execute program and get reward is result is same as the label
                exe = execute(prog.token_seq,sample.structured_rep,logical_tokens_mapping)
                if exe is None:
                    exe = True
                    empty_beam += 1
                execution_results.append(exe)

            if not beam:
                execution_results.append(True)
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

    return

TRAINED_WEIGHTS_SUPERVISED = os.path.join(definitions.ROOT_DIR, 'seq2seqModel' ,'learnedWeights','trained_variables2.ckpt')
if __name__ == '__main__':
    with tf.Session() as sess:
        start = time.time()
        run_unsupervised_training(sess, load_params_path= TRAINED_WEIGHTS3, save_model_path= TRAINED_UNS_WEIGHTS)
        finish = time.time()
        print("elapsed time for 30 ephocs: %s" % (time.strftime("%H%M%S", time.localtime(finish - start))))
        data = CNLVRDataSet(definitions.TRAIN_JSON, ignore_all_true=False)
        run_inference(sess, data, load_params=TRAINED_UNS_WEIGHTS )
    print("done")