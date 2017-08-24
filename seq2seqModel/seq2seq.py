import tensorflow as tf
#from tensorflow.nn.rnn_cell import BasicLSTMCell
from tensorflow.contrib.rnn import BasicLSTMCell
from seq2seqModel.utils import epsilon_greedy_sample, execute
import operator
from seq2seqModel.logical_forms_generator import *
#from word2vec import *
from handle_data import *
import pickle
import numpy as np

# hyperparameters
words_embedding_size = 8
logical_tokens_embedding_size = 10
d1 = 50
hidden_layer_size = 30
sent_embedding_size = 2 * hidden_layer_size
max_sent_length = 5
learning_rate = 0.001
beta = 0.5
epsilon = 0.05
num_of_steps = 10000
max_beam_steps = 30
beam_size = 50
batch_size = 8
STACK = 0
history_length = 4
history_embedding_size = history_length * logical_tokens_embedding_size


# build data

# load data
train = definitions.TRAIN_JSON
TOKEN_MAPPING = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping')
data = read_data(train)
samples, _ = build_data(data, preprocessing_type='lemmatize')
token_mapping = load_functions(TOKEN_MAPPING)
n_logical_tokens = len(token_mapping)



# load word embeddings
embeddings_file = open('word_embeddings','rb')
embeddings_dict = pickle.load(embeddings_file)
embeddings_file.close()

# create sentences, images and label vectors
embedded_sentences, images, labels, lengths = [], [], [], []
for sample in samples:
    sent = []
    for word in sample.sentence.split():
        if word not in embeddings_dict:
            #print(sample.sentence, word) # this shouldn't happen if we do preprocessing
            sent.append(embeddings_dict['<UNK>'])
        else:
            sent.append(embeddings_dict[word])

    embedded_sentences.append(sent)
    images.append(sample.structured_rep)
    labels.append(sample.label)
    if len(sent) > max_sent_length:
        max_sent_length = len(sent)
    lengths.append(len(sent))


# logical forms embeddings
# TODO
token_mapping = load_functions(TOKEN_MAPPING)
num_of_tokens = len(token_mapping)


# Encoder
# placeholders for sentence and it's length
sentence_placeholder = tf.placeholder(shape = [None, None,words_embedding_size],dtype = tf.float32,name = "sentence_placeholder")
sent_lengths = tf.placeholder(dtype = tf.int32,name = "sent_length_placeholder")

# Forward cell
#lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
lstm_fw_cell = BasicLSTMCell (hidden_layer_size, forget_bias=1.0)
# Backward cell
lstm_bw_cell = BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
# stack cells together in RNN
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, sentence_placeholder,sent_lengths,dtype=tf.float32)
#    outputs: A tuple (output_fw, output_bw) containing the forward and
#   the backward rnn output `Tensor`.
#    output_fw will be a `Tensor` shaped:
#    `[batch_size, max_time, cell_fw.output_size]`


# outputs is a (output_forward,output_backwards) tuple. concat them together to receive h vector
h = tf.concat(outputs,2)    # shape: [batch_size, max_time, 2 * hidden_layer_size ]
# the final utterance is the last output

e_m = h[0,-1,:] # len: 2*hidden_layer_size todo check if this is really what we need

# History embedding
def get_history_embedding(history, STACK=0):
    if STACK:
        # TODO
        raise NotImplementedError
    else:
        # TOKEN implementation : concat #history_length last tokens.
        # if the current history is shorter than #history_length, pad with zero vectors
        if len(history) < history_length:
            result = []
            diff = history_length - len(history)
            for i in range(history_length):
                if i < diff:
                    result = np.concatenate([result, np.zeros([logical_tokens_embedding_size])], 0)
            if len(history) != 0:
                result = np.concatenate([result, history], 0)
            return result
        result = []
        for i in range(1, history_length+1):
            result = np.concatenate([result,history[-i]], 0)
        return result

# Decoder
history_embedding = tf.placeholder(shape=[history_embedding_size], dtype = tf.float32, name ="history_embedding")
decoder_input = tf.reshape(tf.concat([e_m,history_embedding],0),[sent_embedding_size + history_embedding_size,1])
W_q = tf.get_variable("W_q", shape=[d1, sent_embedding_size + history_embedding_size], initializer=tf.contrib.layers.xavier_initializer())# (d1,100)
q_t = tf.nn.relu(tf.matmul(W_q, decoder_input)) # dim [d1,1]
W_a = tf.get_variable("W_a",shape=[d1, sent_embedding_size],initializer=tf.contrib.layers.xavier_initializer()) # dim [d1*60]
alpha = tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(q_t),W_a),tf.transpose(h[0]))) #dim [1,25]
c_t = tf.matmul(alpha,h[0,:,:]) # attention vector, dim [1,60]
token_embeddings =  tf.get_variable("token_embeddings", shape=[n_logical_tokens, logical_tokens_embedding_size], initializer=tf.contrib.layers.xavier_initializer())
W_s = tf.get_variable("W_s",shape=[logical_tokens_embedding_size, d1 + sent_embedding_size],initializer=tf.contrib.layers.xavier_initializer())

token_prob_dist = tf.nn.softmax(tf.matmul(token_embeddings, tf.matmul(W_s, tf.concat([q_t,tf.transpose(c_t)],0))))

theta = tf.trainable_variables()
program_probs = tf.placeholder(tf.float32,name="program_probs") # the probability (scalar) of every program in beam
chosen = tf.placeholder(name="chosen",dtype=tf.float32) # for every prg in beam, contains sum of logs of token_prob_dist*one_hot_vector representing the chosen token
rewards = tf.placeholder(tf.float32,name="rewards") # the reward of every program in beam

# loss and gradients
logP = chosen
q_mml = (program_probs * rewards) / tf.reduce_sum(program_probs * rewards)
gradient_weights = tf.pow(q_mml,beta) / tf.reduce_sum(tf.pow(q_mml,beta))
loss = -tf.reduce_sum(gradient_weights * rewards * logP)
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
newGrads = adam.compute_gradients(loss)
w1 = tf.placeholder(tf.float32,name="gradbatch1")
b1 = tf.placeholder(tf.float32,name="gradbatch2")
w2 = tf.placeholder(tf.float32,name="gradbatch3")
b2 = tf.placeholder(tf.float32,name="gradbatch4")
wq = tf.placeholder(tf.float32,name="gradbatch5")
wa = tf.placeholder(tf.float32,name="gradbatch6")
ws = tf.placeholder(tf.float32,name="gradbatch7")
print(theta)
batchGrad = [w1,b1,w2,b2,wq,wa,ws]
updateGrads = adam.apply_gradients(zip(batchGrad,theta))

# training

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    batch_counter = 0
    gradList = sess.run(theta)
    gradBuffer = {}

    #initialize gradients
    for var, grad in enumerate(gradList):
        gradBuffer[var] = 0
    while step < num_of_steps:
        x = embedded_sentences[step % len(embedded_sentences)]
        image = images[step % len(embedded_sentences)]
        label = labels[step % len(embedded_sentences)]
        length = lengths[step % len(embedded_sentences)]

        x = np.reshape(x, [1,len(x),words_embedding_size])
        #encoder_output = sess.run(h,feed_dict={sentence_placeholder: x})
        beam_steps = 0
        beam = []
        keepGoing=True
        # create a beam of possible programs for sentence, the iteration continues while there are unfinished programs in beam and t < max_beam_steps
        while keepGoing and beam_steps < max_beam_steps:
            all_options = []

            if beam_steps == 0: # first iteration - beam is empty

                current_history_embedding = get_history_embedding([]) # no previous history embedding
                print(current_history_embedding)
                # run forward pass
                current_probs = sess.run(token_prob_dist, feed_dict={sentence_placeholder: x,
                                                                     sent_lengths: length,
                                                                     history_embedding: current_history_embedding})
                # in the first iteration the beam is empty, so we add all the possible first tokens and their probabilities
                # each elem in beam is a tuple (p1*..*pt, [z1,..,zt],[p1_function,...,pt_function])
                for i in range(len(current_probs)):
                    # using one-hot vector to calculate the log probability of each token as a function
                    one_hot = tf.zeros([num_of_tokens])
                    one_hot[i] = 1
                    beam.append((current_probs[i],[token_embeddings[i]],[tf.log(token_prob_dist)*one_hot]))
                # if token_num > beam_size and we need to crop the beam
                if beam_size < len(current_probs):
                    beam = epsilon_greedy_sample(beam.sort(key=operator.itemgetter(0)),beam_size,epsilon)

            else: # there are already some programs in the beam

                for prog in beam: # each elem in beam is a tuple (p1*..*pt, [z1,..,zt],[p1_function,...,pt_function])

                    # if current program is finished, add it to options list and proceed to the next one
                    if prog[1][-1] == '<EOS>': #TODO change to vector
                        all_options.append(prog)
                        continue

                    # get history embedding
                    current_history_embedding = get_history_embedding(prog[1])
                    # run forward pass
                    current_probs = sess.run(token_prob_dist, feed_dict={sentence_placeholder: x,
                                                                         sent_lengths: length,
                                                                         history_embedding: current_history_embedding})
                    # for every possible token, calculate probability of choosing it and add to all_options list
                    for i in range(len(current_probs)):
                        # using one-hot vector to calculate the log probability of each token as a function
                        one_hot= tf.zeros([num_of_tokens])
                        one_hot[i] = 1
                        all_options.append((prog[0]*current_probs[i],prog[1].append(token_embeddings[i]),prog[2].append(tf.log(token_prob_dist)*one_hot)))
                # choose the #beam_size programs and place them in the beam
                beam = epsilon_greedy_sample(all_options.sort(key=operator.itemgetter(0)),beam_size,epsilon)
                keepGoing=False
                # continue iterating if there is unfinished program in the beam
                for prog in beam:
                    if prog[1][-1]!='<EOS>':  #TODO change to vector
                        keepGoing=True
            beam_steps+=1

        # calculate rewards and gather probabilities for beam
        current_rewards, probabilities, p_functions = [], [], []
        for prog in beam:
            # execute program and get reward is result is same as the label
            current_rewards.append(bool(execute(prog[1],image,token_mapping)==label))
            probabilities.append(prog[0])
            p_functions.append(tf.reduce_sum(prog[2])) # log of product is sum of logs


        # calculate current gradient
        tGrad, current_loss = sess.run([newGrads,loss],feed_dict={program_probs: probabilities, rewards: current_rewards, chosen: p_functions})
        for var,grad in enumerate(tGrad):
            gradBuffer[var]+=grad[0]

        # update gradients per batch
        if step % batch_size == 0 :
            upGrads = sess.run(updateGrads, feed_dict={w1: gradBuffer[theta[0]],b1: gradBuffer[theta[1]],w2: gradBuffer[theta[2]],
                                                       b2: gradBuffer[theta[3]],wq: gradBuffer[theta[4]],wa: gradBuffer[theta[5]],ws: gradBuffer[theta[6]]})
            for var, grad in enumerate(gradBuffer):
                gradBuffer[var] = 0

            print("step: %d, loss: %2.f" % (step, loss))
        step+=1