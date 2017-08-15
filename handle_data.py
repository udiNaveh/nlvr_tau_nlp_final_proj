import json

from structured_rep_utils import *
from definitions import *
from preprocessing import preprocess_sentences



train = TRAIN_JSON


def read_data(filename):
    data = []
    with open(filename) as data_file:
        for line in data_file:
            data.append(json.loads(line))
    return data


def build_data(data, preprocess = False):
    '''
    
    :param data: a deserialized version of a dataset json file: List[List[List[Dict[str: str]]]]
    :param preprocess: if True, sentences read from the data are preprocessed for spelling correction etc.
    :return: 
    samples : a list of Sample objects (see structured_rep.py). Each represents in a convenient, OOP way 
    a single line from the data
    sentences : a dictionary that maps sentence ids to the sentences, where each unique sentence appears only once
    '''

    samples = []
    sentences = {}
    for line in data:
        samples.append(Sample(line))
        s_index = int(str.split(line["identifier"], "-")[0])
        if s_index not in sentences:
            sentences[s_index] = line["sentence"]

    if preprocess:
        sentences = preprocess_sentences(sentences)

    #for s in samples: # I deleted this part because it puts the same sentence in all of the samples
    #    s_index = int(str.split(line["identifier"], "-")[0])
    #    s.sentence = sentences[s_index]

    return samples, sentences
