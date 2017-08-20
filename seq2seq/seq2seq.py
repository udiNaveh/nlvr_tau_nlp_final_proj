import tensorflow as tf
from utils import *
import operator
from logical_forms_generator import *
from sandbox\omers_word2vec import *

# hyperparameters
words_embedding_size = 20
logical_tokens_embedding_size = 10
hidden_layer_size = 30
#vocab_size =
max_sent_length =
learning_rate = 0.001
beta = 0.5
epsilon = 0.05
num_of_steps = 10000
max_beam_steps = 30
beam_size = 50
batch_size = 8
STACK = 0
history_length = 4

# word embeddings
#TODO

# logical forms embeddings
# TODO

sentence_placeholder = tf.placeholder(shape = [None, max_sent_length,words_embedding_size],dtype = tf.int32,name = "sentence_placeholder")
sent_lengths = tf.placeholder(shape = [None],dtype = tf.int32,name = "sent_length_placeholder")

# Encoder
# Forward cell
lstm_fw_cell = tf.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
# Backward cell
lstm_bw_cell = tf.rnn.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
#unstack to match required input shape: T lists of tensors, each of shape [batch_size, input_size]
unstacked_sentences = tf.unstack(sentence_placeholder, max_sent_length, 1)
outputs, _ = tf.rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, unstacked_sentences,sent_lengths,dtype=tf.float32)
e_m = tf.concat(outputs[0][-1],outputs[1][0])
h = tf.concat([outputs[0],outputs[1]],1)
#e_m = h[-1]

# History embedding
def get_history_embedding(history, STACK=0):
    if len(history) == 0:
        return ''#TODO what?
    if STACK:
        # TODO
    else:
        result=[]
        for i in range(1,history_length+1):
            result=tf.concat([result,history[-i]])
        return result

# Decoder #TODO shapes
history_embedding = tf.placeholder(shape = [],dtype = tf.int32,name = "history_embedding")
token_embeddings = tf.placeholder(shape = [],dtype = tf.int32,name = "token_embeddings")
#e_m = tf.placeholder(shape = [],dtype = tf.int32,name = "e_m")
#h = tf.placeholder(shape = [],dtype = tf.int32,name = "h")
W_q = tf.get_variable("W_q",shape=[hidden_layer_size*2,hidden_layer_size*2],initializer=tf.contrib.layers.xavier_initializer())
q_t = tf.nn.relu(tf.matmul(W_q,tf.concat(e_m,history_embedding)))
W_a = tf.get_variable("W_a",shape=[hidden_layer_size*2,hidden_layer_size*2],initializer=tf.contrib.layers.xavier_initializer())
alpha = tf.nn.softmax(tf.matmul(q_t.T,W_a)*h)
c_t = tf.dot(alpha,h) # attention vector
W_s = tf.get_variable("W_s",shape=[hidden_layer_size*2,logical_tokens_embedding_size],initializer=tf.contrib.layers.xavier_initializer())
token_prob_dist = tf.nn.softmax(tf.matmul(token_embeddings.T,W_s)*tf.concat(q_t,c_t))

theta = tf.trainable_variables()
program_probs = tf.placeholder(tf.float32,name="program_probs")
rewards = tf.placeholder(tf.float32,name="rewards")

# loss and gradients
logP = tf.log(program_probs)
q_mml = (program_probs * rewards) / tf.reduce_sum(program_probs * rewards)
gradient_weights = tf.pow(q_mml,beta) / tf.reduce_sum(tf.pow(q_mml,beta))
loss = -tf.reduce_sum(tf.reduce_sum(gradient_weights * rewards * logP))
#newGrads = tf.gradients(loss,theta)
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
newGrads = adam.compute_gradients(loss)
#batchGrad = {}
batchGrad = tf.placeholder(tf.float32,name="batchGrad")
updateGrads = adam.apply_gradients(batchGrad)#zip(batchGrad,theta)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step=1
    batch_counter = 0
    gradList = sess.run(theta)
    gradBuffer = {}
    token_mapping = load_functions(TOKEN_MAPPING)
    for var, grad in gradList:
        gradBuffer[var] = 0
    while step < num_of_steps:
        x = embedded_sentences[step % len(embedded_sentences)]
        image = images[step % len(embedded_sentences)]
        #TODO reshape
        #encoder_output = sess.run(h,feed_dict={sentence_placeholder: x})
        beam_steps=0
        beam = []
        keepGoing=True
        # create a beam of possible programs for sentence
        while keepGoing and beam_steps < max_beam_steps:
            all_options = []
            if beam_steps == 0: # first iteration - beam is empty
                current_history_embedding = get_history_embedding([]) # no history embedding
                current_probs = sess.run(token_prob_dist, feed_dict={sentence_placeholder: x,
                                                                     history_embedding: current_history_embedding})
                for i in range(len(current_probs)):
                    beam.append((current_probs[i],[token_embeddings[i]]))
                if beam_size < len(current_probs):
                    beam = epsilon_greedy_sample(beam.sort(key=operator.itemgetter(0)),beam_size,epsilon)
            else:
                for prog in beam: # each elem in beam is a tuple (p, [z1,..,zt])
                    current_history_embedding = get_history_embedding(prog[1])
                    current_probs = sess.run(token_prob_dist, feed_dict={sentence_placeholder: x,
                                                                         history_embedding: current_history_embedding})
                    for i in range(len(current_probs)):
                        all_options.append((prog[0]*current_probs[i],prog[1].append(token_embeddings[i])))
                beam = epsilon_greedy_sample(all_options.sort(key=operator.itemgetter(0)),beam_size,epsilon)
                keepGoing=False
                for prog in beam:
                    if prog[1][-1]!='<EOS>': # if there is unfinished program in the beam #TODO change to vector
                        keepGoing=True
            beam_steps+=1

        # calculate rewards and gather probabilities for beam
        current_rewards, probabilities = [], []
        for prog in beam:
            current_rewards.append(execute(prog[1],image,token_mapping))
            probabilities.append(prog[0])

        # calculate current gradient
        tGrad, current_loss = sess.run([newGrads,loss],feed_dict={program_probs: probabilities, rewards: current_rewards})
        for var,grad in tGrad:
            gradBuffer[var]+=grad

        # update gradients per batch
        if step % batch_size == 0 :
            upGrads = sess.run(updateGrads, feed_dict=gradBuffer)
            for var, grad in gradBuffer:
                gradBuffer[var] = 0

            print("step: %d, loss: %2.f" % (step, loss))
        step+=1