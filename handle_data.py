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


def rewrite_data(filename, data, mapping):
    with open(filename, 'w') as output_file:
        for sample in data:
            s_index = int(str.split(sample["identifier"], "-")[0])
            sample["sentence"] = mapping[s_index]
            line = json.dump(sample,output_file)
            output_file.write('\n')


            #output_file.writelines(data)

    return



def build_data(data, preprocessing_type = None):
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

    sentences = preprocess_sentences(sentences, mode=None, processing_type=preprocessing_type)

    for s in samples:
       s_index = int(str.split(s.identifier, "-")[0])
       s.sentence = sentences[s_index]

    return samples, sentences


if __name__ == '__main__':
    data = read_data(TRAIN_JSON)
    samples, sentences = build_data(data, preprocessing_type='thorough')
    rewrite_data(str.replace(TRAIN_JSON, "train.", "train_preprocessed."), data, sentences)
