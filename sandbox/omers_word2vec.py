import numpy as np
import tensorflow as tf
from random import shuffle
import json
import pickle
import definitions
from handle_data import *


sents_path = None
EMBED_DIM = 20
lr = 0.1
ITERNUM = 3

def create_dict(sents_path):
    words_list = []
    with open(sents_path) as sents:
        for sent in sents:
            words = sent.rstrip().split()
            for word in words:
                if word in words_list:
                    pass
                else:
                    words_list.append(word)
    return words_list

def create_dict_from_list(sents_list):
    words_list = []
    for sent in sents_list:
        words = sent.rstrip().split()
        for word in words:
            if word in words_list:
                pass
            else:
                words_list.append(word)
    return words_list


def index_to_one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1.
    return vec

def convert_words_to_indices(sents_list,words_list):
    newsents = []
    #with open(sents_path) as sents:
    assert type(sents_list) == str
    for sent in sents_list:
        newsent = []
        for word in sent.rstrip().split():
            newsent.append(words_list.index(word))
        newsents.append(newsent)
    return newsents

def get_env(k, sent):
    if k == 0:
        env = [sent[1], sent[2]]
    elif k == 1:
        env = [sent[0], sent[2], sent[3]]
    elif k == len(sent) - 1:
        env = [sent[k - 2], sent[k - 1]]
    elif k == len(sent) - 2:
        env = [sent[k - 2], sent[k - 1], sent[k + 1]]
    else:
        env = [sent[k - 2], sent[k - 1], sent[k + 1], sent[k + 2]]
    return env

def word2vec(sents_path, embed_dim = EMBED_DIM, path = False):

    if path:
        words_list = create_dict(sents_path)
    else:
        words_list =create_dict_from_list(sents_path)

    onehots = tf.placeholder(tf.float32, [None, len(words_list)])
    Win = tf.get_variable("Win", shape=[len(words_list), embed_dim], initializer=tf.random_uniform_initializer(0, 1))
    h = tf.nn.tanh(tf.matmul(onehots, Win))
    Wout = tf.get_variable("Wout", shape=[embed_dim, len(words_list)], initializer=tf.random_uniform_initializer(0, 1))
    z = tf.matmul(h, Wout)
    # yt = tf.placeholder(tf.float32, [None, len(words_list)])
    yt = tf.placeholder(tf.float32, [None, 1])

    # yaxol lihiyot shehu mecape levector im indexim velo le-one-hot-im
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = yt, logits = z)

    opt = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    newsents = convert_words_to_indices(sents_path,words_list)

    indices = range(len(newsents))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(ITERNUM):
            shuffle(indices)
            for j in range(len(newsents)):
                currsent = newsents[indices[j]]
                for k, word in enumerate(currsent):
                    env = get_env(k, currsent)
                    envvec = np.sum([index_to_one_hot(l, size= (len(words_list))) for l in env])
                    sess.run(opt, feed_dict = {onehots: envvec, yt: word})
        embeds = sess.run(Win)

    embed_dict = {}
    for i, embed in enumerate(embeds):
        embed_dict[words_list[i]] = embed

    #wh = open('word_embeddings.json', 'w')
    #json.dump(embed_dict, wh)
    #return embed_dict
    file = open('word_embeddings','wb')
    pickle.dump(embed_dict,file)
    file.close()


train = definitions.TRAIN_JSON
data = read_data(train)
samples, sents_dict = build_data(data, preprocessing_type='lemmatize')
sents_to_parse = sents_dict.values()
word2vec(sents_to_parse)