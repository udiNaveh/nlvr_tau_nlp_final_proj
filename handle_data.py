import json
from itertools import permutations
from structured_rep_utils import *
import definitions
import numpy as np
from preprocessing import preprocess_sentences, replace_rare_words_with_unk, get_ngrams_counts, write_ngrams



train = definitions.TRAIN_JSON


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



def build_data(data, preprocessing_type = None, use_unk = True):
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
    if use_unk:
        sentences = replace_rare_words_with_unk(sentences)

    for s in samples:
       s_index = int(str.split(s.identifier, "-")[0])
       s.sentence = sentences[s_index]

    return samples, sentences


class CNLVRDataSet:

    def __init__(self, path):
        self.original_sentences = {}
        self.processed_sentences = {}
        self.samples = {}
        self.__ids = []
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.__num_examples = 0
        self.__ids_by_complexity = []
        self.get_data(path)

    def get_data(self, path):
        data = read_data(path)
        sentences = {}
        for line in data:
            self.samples[line["identifier"]] = Sample(line)
            s_index = int(str.split(line["identifier"], "-")[0])
            if s_index not in sentences:
                sentences[s_index] = line["sentence"]

        self.original_sentences = preprocess_sentences(sentences, processing_type='shallow')
        self.processed_sentences = \
            replace_rare_words_with_unk(preprocess_sentences(sentences, mode=None, processing_type='deep'))

        for s in self.samples.values():
            s_index = int(str.split(s.identifier, "-")[0])
            s.sentence = self.processed_sentences[s_index]

        self.__ids = [k for k in self.original_sentences.keys()]
        self.__num_examples = len(self.__ids)

    def get_samples_by_sentence_id(self, sentence_id):
        samples_ids = ["{0}-{1}".format(sentence_id, i) for i in range(4)]
        return [self.samples[sample_id] for sample_id in samples_ids if sample_id in self.samples]


    def sort_sentences_by_complexity(self, n_classes):
        '''
        mock implementation : sort by length
        '''
        self.__ids_by_complexity = []
        ids_sorted_by_sentence_length = sorted(self.processed_sentences.keys(), key=
                                                    lambda key : len(self.processed_sentences[key].split()))
        class_size = len(self.processed_sentences) // n_classes
        for i in range(n_classes):
            self.__ids_by_complexity.append(ids_sorted_by_sentence_length[
                                            class_size*i : max(class_size*i + class_size, len(self.processed_sentences))])
        return

    def choose_levels_for_curriculum_learning(self, levels):
        if levels is None:
            self.__ids = [k for k in self.original_sentences.keys()]

        self.__ids = [idx for idx in set(ind for level in levels for ind in self.__ids_by_complexity[level])]
        self.__num_examples = len(self.__ids)




    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start == 0:
            np.random.shuffle(self.__ids)  # shuffle index

        # go to the next batch
        elif start + batch_size > self.__num_examples:
            self.epochs_completed += 1
            self._index_in_epoch=0
            return  self.next_batch(batch_size)


        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        indices = self.__ids[start : end]
        return [(self.processed_sentences[k], self.get_samples_by_sentence_id(k)) for k in indices]


def generate_new_samples(dataset, sentence_id):
    sentence = dataset.processed_sentences[sentence_id]
    samples = dataset.get_samples_by_sentence_id(sentence_id)
    color_words = ['yellow', 'blue', 'black']
    tokenized = sentence.split()
    permuts = [p for p in permutations(color_words, 3)[1:]]
    for p in permuts:
        per_s = [p[color_words.index(w)] if w in color_words else w for w in tokenized]













if __name__ == '__main__':
    #data = read_data(train)
    #samples, sentences = build_data(data, preprocessing_type='deep') # check different processig types
    train = CNLVRDataSet(definitions.TRAIN_JSON)
    for i in range(3):
        batch = train.next_batch(8)
    train.sort_sentences_by_complexity(5)
    train.choose_levels_for_curriculum_learning([0])
    for i in range(3):
        batch = train.next_batch(8)
    train.choose_levels_for_curriculum_learning((3,4))
    for i in range(3):
        batch = train.next_batch(12)
    print("")
