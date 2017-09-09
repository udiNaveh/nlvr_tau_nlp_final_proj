import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
import pickle
import numpy as np
import os
import sys
import definitions
from seq2seqModel.utils import *
from seq2seqModel.logical_forms_generation import *
from handle_data import CNLVRDataSet, SupervisedParsing, DataSet
from seq2seqModel.beam import *
from seq2seqModel.hyper_params import *
from general_utils import increment_count, union_dicts
import time




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
words_to_tokens = pickle.load(open(os.path.join(definitions.DATA_DIR, 'logical forms', 'words_to_tokens'), 'rb'))
# words_to_tokens = words_to_tokens_dict(logical_tokens_mapping)
# dic ={}
# with open('words_to_tokens.txt', 'r') as f:
#
#     for line in f:
#         l = line.split(':')
#         k = l[0].strip()
#         v = []
#         for s in l[1:]:
#             v.append(s.strip().split())
#         dic[k] = v
#
# pickle.dump(dic, open(os.path.join(definitions.DATA_DIR, 'logical forms', 'words_to_tokens'), 'wb'))


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


def get_next_token_probs_from_nn(partial_program, logical_tokens_embeddings_dict, decoder_feed_dict,
                                 history_embedding_tensor, token_prob_dist):

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
    decoder_feed_dict[history_embedding_tensor] = history_embs
    current_probs = np.squeeze(sess.run(token_prob_dist, feed_dict=decoder_feed_dict))
    current_probs = np.where(current_probs>0, current_probs, 1e-30)
    if (np.count_nonzero(current_probs) != len(current_probs)):
        print("zero prob")

    probs_given_valid = [1.0] if len(valid_next_tokens) == 1 else \
        [current_probs[logical_tokens_ids[next_tok]] for next_tok in valid_next_tokens]
    probs_given_valid = probs_given_valid / np.sum(probs_given_valid)
    return valid_next_tokens, probs_given_valid


def get_gradient_weights_for_programs(beam_rewarded_programs):
    if not beam_rewarded_programs:
        return []
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



def run_unsupervised(sess, train_dataset, mode, validation_dataset = None, load_params_path = None, save_model_path = None):


    modes = ('train', 'test')
    assert mode in modes
    test_between_training_epochs = validation_dataset is not None
    assert not (mode=='test' and test_between_training_epochs)

    # initialization
    theta = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    compute_program_grads = optimizer.compute_gradients(cross_entropy)
    batch_grad = build_batchGrad()
    update_grads = optimizer.apply_gradients(zip(batch_grad, theta))
    #sess.run(tf.global_variables_initializer())

    # load pre-trained variables, if given
    if load_params_path:
        #tf.train.Saver(var_list=theta).restore(sess, load_params_path)
        tf.train.Saver().restore(sess, load_params_path)
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
    timer = []
    epochs_completed, num_have_pattern, iter, n_samples = 0, 0, 0, 0
    batch_num_in_epoch = 1

    # a dictionary for cached programs
    if USE_CACHED_PROGRAMS and LOAD_CACHED_PROGRAMS and mode=='train':
        cpf = open(CACHED_PROGRAMS, 'rb')
        all_cached_programs = pickle.load(cpf)
    else:
        all_cached_programs = {}


    start = time.time()

    curr_dataset, other_dataset = train_dataset, validation_dataset
    curr_mode = mode
    other_mode = 'train' if mode=='test' else 'test'



    while train_dataset.epochs_completed < n_epochs and not stopping_criterion_met:

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
                # the rating is based on previous rewards of these programs om sentences that
                # are similar in pattern to the current sentence.
                decodings_from_cache = get_programs_for_sentence_by_pattern(sentence, all_cached_programs)
                decodings_from_cache = [token_seq.split() for token_seq in decodings_from_cache][:N_CACHED_PROGRAMS]
                num_have_pattern += 1 if decodings_from_cache else 0

            # create feed dicts for running the current sentence in the nn.
            # running the sentence through the encoder lstm to obtain its output is also done inside the following call.
            # This time-consuming operation is executed only once, and doesn't need to be
            # repeated at each step during the beam search.
            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, (h, e_m))

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
            # perform beam search to obtain candidate parsings for the sentence
            original_sentence_for_beam = sentence if SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH else None

            beam = e_greedy_randomized_beam_search(next_token_probs_getter,
                                                   logical_tokens_mapping,
                                                   original_sentence= original_sentence_for_beam,
                                                   epsilon=epsilon_for_beam,
                                                   suggested_decodings=decodings_from_cache)

            programs_execution_results = {}
            programs_from_cache = []

            for prog in beam:
                if prog.token_seq in decodings_from_cache:
                    programs_from_cache.append(prog)

                programs_execution_results[prog] = get_program_execution_stats(
                    prog.token_seq, related_samples, logical_tokens_mapping)

            if USE_CACHED_PROGRAMS and curr_mode =='train':
                for prog, prog_stats in programs_execution_results.items():
                    if prog_stats.is_consistent or prog in programs_from_cache:
                        update_programs_cache(all_cached_programs, sentence, prog, prog_stats)


            # prog_str = " ".join(prog.token_seq)
            # if sentence_id not in all_consistent_decodings:
            #     all_consistent_decodings[sentence_id] = {}
            # increment_count(all_consistent_decodings[sentence_id], prog_str)

            consistent_programs = [prog for prog, stats in programs_execution_results.items() if stats.is_consistent]

            if curr_mode =='train':
                # in order to mitigate the problem of spurious programs, a program for a given sentence gets
                #  a reward iff it is consistent with all its related images, (i.e. compiles and returns the correct
                # label on all samples related to the sentence).
                # here we take the gradient w.r.t each such consistent program, and weight those gradients
                # using the beta-meritocratic approach presented in Goo at el. 2017

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
                mean_program_lengths.append(np.mean([len(pp) for pp in beam]))
                beam_reranked = programs_reranker_2(sentence, beam, words_to_tokens)
                top_program_by_model = beam[0]
                top_program_by_reranking = beam_reranked[0]
                top_by_model_stats = programs_execution_results[top_program_by_model]
                top_by_reranking_stats = programs_execution_results[top_program_by_reranking]

            if not beam: # no valid program was found by the beam - default is to guess 'True' for all images
                top_by_model_stats = top_by_reranking_stats = get_program_execution_stats(
                    ["ERR"], related_samples, logical_tokens_mapping)

            n_consistent_top_by_model.append(top_by_model_stats.is_consistent)
            n_consistent_top_by_reranking.append(top_by_reranking_stats.is_consistent)
            n_ccorrect_top_by_model.append(top_by_model_stats.n_correct)
            n_correct_top_by_reranking.append(top_by_reranking_stats.n_correct)

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
                    print("parameters used in learning:")
                    for param_name in ['EPSILON_FOR_BEAM_SEARCH', 'BETA', 'SKIP_AUTO_TOKENS', 'N_CACHED_PROGRAMS',
                                       'INJECT_TO_BEAM',
                                        'SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH',
                                       'AVOID_ALL_TRUE_SENTENCES', 'INPUT_WEIGHTS',
                                       'OUTPUT_WEIGHTS']:
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
                    saver2.save(sess, save_model_path, global_step=train_dataset.epochs_completed,write_meta_graph=False)
                    print("saved epoch %d" % train_dataset.epochs_completed)

                if SAVE_CACHED_PROGRAMS:
                    cpf = open(CACHED_PROGRAMS, 'wb')
                    pickle.dump(all_cached_programs, cpf)

        batch_num_in_epoch += 1

        if is_last_batch_in_epoch and test_between_training_epochs:
            curr_mode, other_mode = other_mode, curr_mode
            curr_dataset, other_dataset = other_dataset, curr_dataset

    return


def run_supervised_training(sess, load_params_path = None, save_params_path = None):

    theta = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    compute_program_grads = optimizer.compute_gradients(cross_entropy)
    batch_grad = build_batchGrad()
    update_grads = optimizer.apply_gradients(zip(batch_grad, theta))

    init = tf.global_variables_initializer()
    sess.run(init)
    #saver = tf.train.Saver(theta)
    saver = tf.train.Saver()
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

    while epoch_num < 1:
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
        if current_data_set is train:
            sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})


        for var, grad in enumerate(gradBuffer):
            gradBuffer[var] = gradBuffer[var]*0


    if save_params_path:
        saver.save(sess, save_params_path)

def run_inference(sess, data, load_params, clf=None):

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

    #beam_training_set = pd.DataFrame(columns=[col for col in range(len(features_vec)+1)]) ### TODO create dataframe

    beam_by_sent ={}
    for sample in data.samples.values():
        if total % 10 == 0:
            print("sample %d out of %d" % (total,samples_num))

        sentences = [sample.sentence]
        label = sample.label

        for step in range (len(sentences)):

            sentence = (sentences[step])

            encoder_feed_dict, decoder_feed_dict = \
                get_feed_dicts_from_sentence(sentence, sentence_placeholder, sent_lengths_placeholder, (h, e_m))

            next_token_probs_getter = lambda pp :  get_next_token_probs_from_nn(pp, logical_tokens_embeddings_dict,
                                                                                decoder_feed_dict,
                                                                                history_embedding_placeholder,
                                                                                token_prob_dist_tensor)

            if sentence in beam_by_sent:
                beam = beam_by_sent[sentence]
            else:
                beam = e_greedy_randomized_beam_search(next_token_probs_getter, logical_tokens_mapping,
                                                       original_sentence=sentence, epsilon=0.0)
                beam_by_sent[sentence] = beam

            execution_results = []

            for prog_rank, prog in enumerate(beam):
                # execute program and get reward is result is same as the label
                exe = execute(prog.token_seq,sample.structured_rep,logical_tokens_mapping)

                #if not clf:
                    # TODO get feature vector and append label to it
                    #beam_label = 1 if exe is True else 0
                    #row = feature_vec.append(beam_label)
                    #beam_training_set.append(row,axis=0)



                if exe is None:
                    exe = True
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

    #return beam_training_set
    return


#def run_beam_classifier(beam_training_set):
    # TODO






if __name__ == '__main__':

    orig_stdout = sys.stdout
    STATS_FILE = os.path.join(SEQ2SEQ_DIR, 'training logs', 'stats_' + time.strftime("%Y-%m-%d_%H_%M") + '.txt')
    OUTPUT_WEIGHTS = os.path.join(SEQ2SEQ_DIR, 'learnedWeightsUns', 'weights_' + time.strftime("%Y-%m-%d_%H_%M")+ '.ckpt')
    #INPUT_WEIGHTS = os.path.join(SEQ2SEQ_DIR, 'learnedWeightsUns' , 'weights_2017-09-08_15_47.ckpt-13')
    #INPUT_WEIGHTS = TRAINED_WEIGHTS_UNSUP_HISTORY_4
    train_dataset = CNLVRDataSet(DataSet.TRAIN)
    dev_dataset = CNLVRDataSet(DataSet.DEV)
    test_dataset = CNLVRDataSet(DataSet.TEST)





    # TODO DELETE ALL THE BULLSHIT BEFORE COMMIT



    if AVOID_ALL_TRUE_SENTENCES:
        train_dataset.ignore_all_true_samples()

    with tf.Session() as sess:

        run_unsupervised(sess, train_dataset, mode='test',
                         load_params_path=INPUT_WEIGHTS, save_model_path=OUTPUT_WEIGHTS)
        #run_supervised_training(sess, load_params_path=None, save_params_path=OUTPUT_WEIGHTS)
        # INPUT_WEIGHTS = os.path.join(SEQ2SEQ_DIR, 'learnedWeightsUns' , 'weights_2017-09-09_16_04.ckpt')
        # run_unsupervised(sess, dev_dataset, mode='train',
        #                  load_params_path=INPUT_WEIGHTS, save_model_path=None)


    print("done")