import tensorflow as tf
from utils import epsilon_greedy_sample, execute
import operator
from logical_forms_generator import *
from word2vec import *
from handle_data import *
import pickle
import numpy as np

# hyperparameters
words_embedding_size = 8
logical_tokens_embedding_size = 10
hidden_layer_size = 30
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

# build data

#load data
train = definitions.TRAIN_JSON
data = read_data(train)
samples, _ = build_data(data, preprocessing_type='lemmatize')
#load word embeddings
embeddings_file = open('word_embeddings','rb')
embeddings_dict = pickle.load(embeddings_file)
embeddings_file.close()

embedded_sentences, images, labels, lengths = [], [], [], []
#create sentences, images and label vectors
for sample in samples:
    sent = []
    for word in sample.sentence.split():
        if word not in embeddings_dict:
            print(sample.sentence, word)
            sent.append(embeddings_dict['<UNK>'])
        else:
            sent.append(embeddings_dict[word])
    #sent.append(embeddings_dict['EOS']) #should be done in preprocessing
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
token_embeddings =

sentence_placeholder = tf.placeholder(shape = [None, None,words_embedding_size],dtype = tf.float32,name = "sentence_placeholder")
sent_lengths = tf.placeholder(dtype = tf.int32,name = "sent_length_placeholder")

# Encoder
# Forward cell
lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
# Backward cell
lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size, forget_bias=1.0)
#unstack to match required input shape: T lists of tensors, each of shape [batch_size, input_size]
#unstacked_sentences = tf.unstack(sentence_placeholder, max_sent_length, 1)
outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, sentence_placeholder,sent_lengths,dtype=tf.float32)
#e_m = tf.concat(outputs[0][-1],outputs[1][0])
#h = tf.concat([outputs[0],outputs[1]],1) #TODO is this the definition of h?
h = tf.concat(outputs,2)
e_m = h[0][-1]

# History embedding
def get_history_embedding(history, STACK=0):
    if STACK:
        # TODO
        raise NotImplementedError
    else:
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
history_embedding = tf.placeholder(shape=[logical_tokens_embedding_size*4],dtype = tf.float32,name = "history_embedding")
#token_embeddings = tf.placeholder(dtype = tf.float32,name = "token_embeddings")
#print(outputs[0].shape, h.shape,e_m.get_shape(), history_embedding.get_shape())
W_q = tf.get_variable("W_q",shape=[hidden_layer_size*2,hidden_layer_size*2 + logical_tokens_embedding_size*4],initializer=tf.contrib.layers.xavier_initializer()) # (60,100)
q_t = tf.nn.relu(tf.matmul(W_q,tf.reshape(tf.concat([e_m,history_embedding],0),[hidden_layer_size*2 + logical_tokens_embedding_size*4,1]))) # dim [60,1]
W_a = tf.get_variable("W_a",shape=[hidden_layer_size*2,hidden_layer_size*2],initializer=tf.contrib.layers.xavier_initializer()) # dim [60*60]
alpha = tf.nn.softmax(tf.matmul(tf.matmul(tf.transpose(q_t),W_a),tf.transpose(h[0]))) #dim [1,25]
c_t = tf.matmul(alpha,h[0]) # attention vector, dim [1,60]
W_s = tf.get_variable("W_s",shape=[hidden_layer_size*2,logical_tokens_embedding_size],initializer=tf.contrib.layers.xavier_initializer())
token_prob_dist = tf.nn.softmax(tf.matmul(tf.transpose(token_embeddings),W_s)*tf.concat([q_t,tf.transpose(c_t)],0))

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
        beam_steps=0
        beam = []
        keepGoing=True
        # create a beam of possible programs for sentence
        while keepGoing and beam_steps < max_beam_steps:
            all_options = []
            if beam_steps == 0: # first iteration - beam is empty
                current_history_embedding = get_history_embedding([]) # no history embedding
                print(current_history_embedding)
                current_probs = sess.run(token_prob_dist, feed_dict={sentence_placeholder: x,
                                                                     sent_lengths: length,
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
            current_rewards.append(bool(execute(prog[1],image,token_mapping)==label))
            probabilities.append(prog[0])

        # calculate current gradient
        tGrad, current_loss = sess.run([newGrads,loss],feed_dict={program_probs: probabilities, rewards: current_rewards})
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