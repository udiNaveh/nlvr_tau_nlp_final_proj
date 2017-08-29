import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from seq2seqModel.utils import *
from seq2seqModel.logical_forms_generator import *
from handle_data import CNLVRDataSet, SupervisedParsing
import pickle
import numpy as np
import time


#paths

LOGICAL_TOKENS_MAPPING_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping')
LOGICAL_TOKENS_EMBEDDINGS_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'logical_tokens_embeddings')
WORD_EMBEDDINGS_PATH = os.path.join(definitions.ROOT_DIR, 'seq2seqModel', 'word_embeddings')
PARSED_EXAMPLES_T = os.path.join(definitions.DATA_DIR, 'parsed sentences', 'parses for check as tokens')

####
###hyperparameters
####

#dimensions
words_embedding_size = 8
logical_tokens_embedding_size = 12
decoder_hidden_layer_size = 50
lstm_hidden_layer_size = 30
sent_embedding_size = 2 * lstm_hidden_layer_size
history_length = 4
history_embedding_size = history_length * logical_tokens_embedding_size

#other hyper parameters
learning_rate = 0.001
beta = 0.5
epsilon = 0.2
num_of_steps = 10000
max_decoding_length = 15
beam_size = 30
batch_size = 8




# load word embeddings
embeddings_file = open(WORD_EMBEDDINGS_PATH,'rb')
embeddings_dict = pickle.load(embeddings_file)
embeddings_file.close()
assert words_embedding_size == np.size(embeddings_dict['blue'])



#load logical tokens inventory
logical_tokens_mapping = load_functions(LOGICAL_TOKENS_MAPPING_PATH)
logical_tokens = [token for token in logical_tokens_mapping.keys()]
for var in "xyzwuv":
    logical_tokens.extend([var, 'lambda_{}_:'.format(var) ])
logical_tokens.extend(['<s>', '<EOS>'])
logical_tokens_ids = {lt: i for i, lt in enumerate(logical_tokens)}
n_logical_tokens = len(logical_tokens_ids)



ngram_p_dict = get_probs_from_file(PARSED_EXAMPLES_T)


def build_sentence_encoder():

    # placeholders for sentence and it's length
    sentence_placeholder = tf.placeholder(shape = [None, None,words_embedding_size],dtype = tf.float32,name = "sentence_placeholder")
    sent_lengths = tf.placeholder(dtype = tf.int32,name = "sent_length_placeholder")

    # Forward cell
    lstm_fw_cell = BasicLSTMCell (lstm_hidden_layer_size, forget_bias=1.0)
    # Backward cell
    lstm_bw_cell = BasicLSTMCell(lstm_hidden_layer_size, forget_bias=1.0)
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
    history_embedding = tf.placeholder(shape=[None, history_embedding_size], dtype=tf.float32, name="history_embedding")
    num_rows = tf.shape(history_embedding)[0]
    e_m_tiled = tf.tile(final_utterance_embedding, ([num_rows, 1]))
    decoder_input = tf.concat([e_m_tiled, history_embedding], axis=1)
    W_q = tf.get_variable("W_q", shape=[decoder_hidden_layer_size, sent_embedding_size + history_embedding_size],
                          initializer=tf.contrib.layers.xavier_initializer())  # (d1,100)
    q_t = tf.nn.relu(tf.matmul(W_q, tf.transpose(decoder_input)))  # dim [d1,1]
    W_a = tf.get_variable("W_a", shape=[decoder_hidden_layer_size, sent_embedding_size],
                          initializer=tf.contrib.layers.xavier_initializer())  # dim [d1*60]
    alpha = tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(q_t), W_a), tf.transpose(lstm_outputs)))  # dim [1,25]
    c_t = tf.matmul(alpha, lstm_outputs)  # attention vector, dim [1,60]
    W_s = tf.get_variable("W_s", shape=[logical_tokens_embedding_size, decoder_hidden_layer_size + sent_embedding_size],
                          initializer=tf.contrib.layers.xavier_initializer())
    W_logical_tokens = tf.get_variable("W_logical_tokens", dtype= tf.float32,
                    shape=[n_logical_tokens, logical_tokens_embedding_size],
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

# PartialProgram


def beam_search(lstm_output_tensors, history_embedding_tensor, encoder_feed_dict, token_prob_dist, logical_tokens_embeddings_dict):

    h, e_m = lstm_output_tensors
    # encoder_output = sess.run(h,feed_dict={sentence_placeholder: x})
    beam = [PartialProgram()]

    sentence_h, sentence_e_m = sess.run(lstm_output_tensors, feed_dict= encoder_feed_dict)
    # create a beam of possible programs for sentence, the iteration continues while there are unfinished programs in beam and t < max_beam_steps

    for t in range(max_decoding_length):
        start = time.time()
        continuations = []
        for partial_program in beam:
            if t > 0 and partial_program[-1] == '<EOS>':
                continuations.append(partial_program)
                continue
            history_tokens = ['<s>' for _ in range(history_length - len(partial_program))] + \
                             partial_program[-history_length:]
            history_embs = [logical_tokens_embeddings_dict[tok] for tok in history_tokens]
            history_embs = np.reshape(np.concatenate(history_embs), [1, history_embedding_size])
            # run forward pass

            current_probs = np.squeeze(sess.run(token_prob_dist, feed_dict={e_m: sentence_e_m,
                                                                            h: sentence_h,
                                                                            history_embedding_tensor: history_embs}))

            valid_next_tokens = partial_program.get_possible_continuations(logical_tokens_mapping)

            probs_given_valid = [1.0] if len(valid_next_tokens) == 1 else\
                [current_probs[logical_tokens_ids[next_tok]] for next_tok in valid_next_tokens]
            #TODO document
            logprob_given_valid = np.log(probs_given_valid/np.sum(probs_given_valid))

            #ngram_probs = get_ngram_probs(history_tokens, ngram_p_dict, valid_next_tokens)

            for i, next_tok in enumerate(valid_next_tokens):
                pp = PartialProgram(partial_program)
                # logprob_given_valid[i]
                pp.add_token(next_tok, + logprob_given_valid[i] , logical_tokens_mapping)
                continuations.append(pp)

        # choose the #beam_size programs and place them in the beam
        continuations.sort(key=lambda c: - c.logprob)
        beam = epsilon_greedy_sample(continuations, beam_size, epsilon)

        if all([prog.token_seq[-1] == '<EOS>' for prog in beam]):
            break  # if we have beam_size full programs, no need to keep searching
    return beam


def get_gradient_weights_for_beam(beam_rewarded_programs):

    beam_log_probs = np.array([prog.logprob for prog in beam_rewarded_programs])
    q_mml = softmax(beam_log_probs)
    return np.power(q_mml,beta) / np.sum(np.power(q_mml,beta))


def sentences_to_embeddings(sentences, enbedding_dict):
    return np.array([[embeddings_dict.get(w, embeddings_dict['<UNK>']) for w in sentence] for sentence in sentences])



def run_unsupervised_training(sess):

    # build the computaional graph:
    # bi-lstm encoder - given a sentence (of a variable length) as a sequence of word embeddings,
    # and returns the lstm outputs.
    sentence_placeholder, sent_lengths_placeholder, h, e_m = build_sentence_encoder()
    # ff decoder - given the outputs of the encoder, and an embedding of the decoding history,
    # computes a probability distribution over the tokens.
    history_embedding, token_unnormalized_dist, W_logical_tokens = build_decoder(h, e_m)
    token_prob_dist = tf.nn.softmax(token_unnormalized_dist,dim=0)
    chosen_logical_tokens = tf.placeholder(tf.float32, [None, n_logical_tokens],  ##
                                   name="chosen_action_token")  # a one-hot vector represents the action taken at each step

    # the log-probability according to the model of a program given an input sentnce.
    program_log_prob = tf.reduce_sum(tf.log(token_prob_dist) * tf.transpose(chosen_logical_tokens))
    theta = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    compute_program_grads = optimizer.compute_gradients(program_log_prob)
    batch_grad = build_batchGrad()
    update_grads = optimizer.apply_gradients(zip(batch_grad, theta))

    init = tf.global_variables_initializer()

    sess.run(init)
    step=1
    gradList = sess.run(theta) # just to get dimensions
    gradBuffer = {}

    # load data
    train = CNLVRDataSet(definitions.TRAIN_JSON)

    #initialize gradients
    for var, grad in enumerate(gradList):
        gradBuffer[var] = grad*0
    while train.epochs_completed < 5:
        print("train.epochs_completed= {}".format(train.epochs_completed))

        sentences, samples = zip(*train.next_batch(batch_size))

        embedded_sentences = sentences_to_embeddings(sentences, embeddings_dict)
        current_logical_tokens_embeddings = sess.run(W_logical_tokens)
        logical_tokens_embeddings_dict = \
            {token : current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens}
        for step in range (batch_size):
            s = sentences[step]
            x = embedded_sentences[step]
            related_samples = samples[step]
            sentence_embedding = np.reshape(x, [1, len(x), words_embedding_size])
            length = [len(s.split())]
            encoder_feed_dict = {sentence_placeholder: sentence_embedding, sent_lengths_placeholder: length}

            beam = beam_search((h, e_m), history_embedding, encoder_feed_dict, token_prob_dist,
                               logical_tokens_embeddings_dict)

            # calculate rewards and gather probabilities for beam
            rewarded_programs = []
            beam = [prog for prog in beam if prog.token_seq[-1] == '<EOS>'] #otherwise won't compile and therefore no reward

            compiled = 0
            correct = 0
            for prog in beam:
                prog.token_seq.pop(-1) # take out the '<EOS>' token
                # execute program and get reward is result is same as the label
                execution_results = np.array([execute(prog.token_seq,sample.structured_rep,logical_tokens_mapping)
                                     for sample in related_samples])
                actual_labels = np.array([sample.label for sample in related_samples])
                compiled+= sum(res is not None for res in execution_results)
                reward = 1 if all(execution_results==actual_labels) else 0
                correct+=reward
                if reward>0:
                    rewarded_programs.append(prog)

            #print("beam size = {0}, {1} programs compiled, {2} correct".format(len(beam), compiled, correct))

            print("beam, compliled, correct = {0} /{1} /{2}".format(len(beam),compiled ,correct))

            if not rewarded_programs:
                continue

            programs_gradient_weights = get_gradient_weights_for_beam(rewarded_programs)

            for idx, program in enumerate(rewarded_programs):
                padded_token_sequence = ['<s>' for _ in range(history_length)] + [token for token in program]
                padded_token_sequence_embedded = np.concatenate([logical_tokens_embeddings_dict[tok] for tok in padded_token_sequence])
                token_sequence_ids = [logical_tokens_ids.get(token, -1) for token in program]
                token_sequence_one_hot = np.stack([one_hot(n_logical_tokens, tok_id) if tok_id >= 0 else np.zeros(n_logical_tokens)
                                                   for tok_id in token_sequence_ids], axis = 0)
                histories = [padded_token_sequence_embedded[logical_tokens_embedding_size * i :
                logical_tokens_embedding_size * i + history_embedding_size] for i in range(len(program))]
                histories = np.reshape(histories, [len(program), history_embedding_size])


                program_grad = sess.run(compute_program_grads, feed_dict={sentence_placeholder: sentence_embedding,
                                                                          sent_lengths_placeholder: length,
                                                                          history_embedding : histories,
                                                                          chosen_logical_tokens : token_sequence_one_hot
                                                                          })
                for var,grad in enumerate(program_grad):
                   gradBuffer[var] -=  programs_gradient_weights[idx] * grad[0]


        sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})
        for var, grad in enumerate(gradBuffer):
            gradBuffer[var] = gradBuffer[var]*0


def run_supervised_training(sess):

    # build the computaional graph:
    # bi-lstm encoder - given a sentence (of a variable length) as a sequence of word embeddings,
    # and returns the lstm outputs.
    sentence_placeholder, sent_lengths_placeholder, h, e_m = build_sentence_encoder()
    # ff decoder - given the outputs of the encoder, and an embedding of the decoding history,
    # computes a probability distribution over the tokens.
    history_embedding, token_unnormalized_dist, W_logical_tokens = build_decoder(h, e_m)
    # a one-hot vector represents the action taken at each step
    chosen_logical_tokens = tf.placeholder(tf.float32, [1, n_logical_tokens],
                                   name="chosen_action_token")

    #chosen_logical_tokens_reshaped = np.reshape(chosen_logical_tokens,[n_logical_tokens,1])
    token_unnormalized_dist = tf.transpose(token_unnormalized_dist)

    # cross-entropy loss per single token in a single sentence
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=chosen_logical_tokens, logits=token_unnormalized_dist))
    theta = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    compute_program_grads = optimizer.compute_gradients(cross_entropy)
    batch_grad = build_batchGrad()
    update_grads = optimizer.apply_gradients(zip(batch_grad, theta))

    init = tf.global_variables_initializer()

    sess.run(init)
    step=1
    gradList = sess.run(theta) # just to get dimensions
    gradBuffer = {}

    # load data
    train = SupervisedParsing(definitions.SUPERVISED_TRAIN_PICKLE)

    #initialize gradients
    for var, grad in enumerate(gradList):
        gradBuffer[var] = grad*0

    steps_num = 0

    while train.epochs_completed < 5:
        print("epoch %d, step %d" % (train.epochs_completed, steps_num))
        #print("train.epochs_completed= {}".format(train.epochs_completed))
        steps_num += batch_size

        sentences, labels = zip(*train.next_batch(batch_size))
        embedded_sentences = sentences_to_embeddings(sentences, embeddings_dict)
        current_logical_tokens_embeddings = sess.run(W_logical_tokens)
        logical_tokens_embeddings_dict = \
            {token : current_logical_tokens_embeddings[logical_tokens_ids[token]] for token in logical_tokens}
        for step in range(batch_size):
            s = sentences[step]
            x = embedded_sentences[step]
            #print(s)
            #print(labels[step])

            sentence_embedding = np.reshape(x, [1, len(x), words_embedding_size])
            length = [len(s.split())]
            encoder_feed_dict = {sentence_placeholder: sentence_embedding, sent_lengths_placeholder: length}
            sentence_h, sentence_e_m = sess.run((h,e_m), feed_dict=encoder_feed_dict)
            golden_parsing = labels[step].split()
            #print(golden_parsing)
            golden_parsing.append('<EOS>')

            output = []

            for i in range(len(golden_parsing)):

                # embed golden history

                history_tokens = ['<s>' for _ in range(history_length - i)] + \
                                 golden_parsing[:i][-history_length:]
                history_embs = [logical_tokens_embeddings_dict[tok] for tok in history_tokens]
                history_embs = np.reshape(np.concatenate(history_embs), [1, history_embedding_size])

                # run forward pass

                current_probs = np.squeeze(sess.run(token_unnormalized_dist, feed_dict={e_m: sentence_e_m,
                                                                                        h: sentence_h,
                                                                                        history_embedding: history_embs}))
                # get one-hot representation of the golden token
                one_hot_vec = np.zeros(n_logical_tokens)
                if golden_parsing[i] == 'and':
                    golden_parsing[i] = 'AND'
                one_hot_vec[logical_tokens_ids[golden_parsing[i]]] = 1
                one_hot_reshaped = np.reshape(one_hot_vec,[1,n_logical_tokens])

                #if train.epochs_completed == 4:
                output.append(logical_tokens[np.argmax(current_probs)])

                # calculate gradient
                token_grad = sess.run(compute_program_grads, feed_dict={sentence_placeholder: sentence_embedding,
                                                                          sent_lengths_placeholder: length,
                                                                          history_embedding : history_embs,
                                                                        chosen_logical_tokens: one_hot_reshaped})

                for var,grad in enumerate(token_grad):
                   gradBuffer[var] -= grad[0]
            #if train.epochs_completed == 4:
            print("outputted: %s" % " ".join(output))

        sess.run(update_grads, feed_dict={g: gradBuffer[i] for i, g in enumerate(batch_grad)})

        for var, grad in enumerate(gradBuffer):
            gradBuffer[var] = gradBuffer[var]*0


if __name__ == '__main__':
    with tf.Session() as sess:
        run_supervised_training(sess)
