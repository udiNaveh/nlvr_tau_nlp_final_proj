import json
from itertools import permutations
from structured_rep import *
import definitions
import numpy as np
from preprocessing import preprocess_sentences, replace_rare_words_with_unk, get_ngrams_counts, write_ngrams
import pickle




class DataSet(Enum):
    TRAIN = 'train',
    DEV = 'dev',
    TEST = 'test'


paths = {   DataSet.TRAIN : definitions.TRAIN_JSON,
            DataSet.DEV : definitions.DEV_JSON,
            DataSet.TEST : definitions.TEST_JSON}


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

    def __init__(self, dataset):
        self.__dataset = dataset
        self.original_sentences = {}
        self.processed_sentences = {}
        self.samples = {}
        self.__ids = []
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.__ids_by_complexity = []
        self.get_data(paths[dataset])

    @property
    def name(self):
        return self.__dataset.name

    @property
    def num_examples(self):
        return len(self.__ids)

    def get_samples_by_sentence_id(self, sentence_id):
        samples_ids = ["{0}-{1}".format(sentence_id, i) for i in range(4)]
        return [self.samples[sample_id] for sample_id in samples_ids if sample_id in self.samples]

    def get_sentence_by_id(self, sentence_id):
        return self.processed_sentences[sentence_id]

    def get_data(self, path):
        data = read_data(path)
        sentences = {}
        for line in data:
            self.samples[line["identifier"]] = Sample(line)
            s_index = int(str.split(line["identifier"], "-")[0])
            if s_index not in sentences:
                sentences[s_index] = line["sentence"]

        self.original_sentences = preprocess_sentences(sentences, processing_type='shallow')

        if self.__dataset == DataSet.TRAIN:
            self.processed_sentences = \
                replace_rare_words_with_unk(preprocess_sentences(sentences, mode=None, processing_type='deep'))

        else:
            self.processed_sentences = \
                replace_rare_words_with_unk(preprocess_sentences(sentences, mode='r', processing_type='deep'),
                                            definitions.TOKEN_COUNTS_PROCESSED)

        for s in self.samples.values():
            s_index = int(str.split(s.identifier, "-")[0])
            s.sentence = self.processed_sentences[s_index]

        self.__ids = [k for k in self.original_sentences.keys()]

    def use_subset_by_sentnce_condition(self, f_s):
        ''' f_id is a boolean function on ids'''
        new_ids = []
        for k, s in self.processed_sentences.items():
            if f_s(s):
                new_ids.append(k)
        self.__ids = new_ids

    def use_subset_by_images_condition(self, f_im):
        ''' f_im is a boolean function on a set of samples'''
        new_ids = []
        for k, s in self.processed_sentences.items():
            related_samples = self.get_samples_by_sentence_id(k)
            if f_im(related_samples):
                new_ids.append(k)
        self.__ids = new_ids

    def ignore_all_true_samples(self):
        all_true_filter = lambda s_samples : not all([s.label==True for s in s_samples])
        self.use_subset_by_images_condition(all_true_filter)

    def sort_sentences_by_complexity(self, complexity_measure, n_classes):
        '''
        mock implementation : sort by length
        '''
        self.__ids_by_complexity = []
        ids_sorted_by_sentence_length = sorted(self.processed_sentences.keys(), key=
                                                    lambda key : complexity_measure(self.processed_sentences[key]))
        class_size = len(self.processed_sentences) // n_classes
        for i in range(n_classes):
            self.__ids_by_complexity.append(ids_sorted_by_sentence_length[
                                            class_size*i : min(class_size*i + class_size, len(self.processed_sentences))])
        return

    def choose_levels_for_curriculum_learning(self, levels):
        self.__ids = [idx for idx in set(ind for level in levels for ind in self.__ids_by_complexity[level])]

    def restart(self):
        self.__ids = [k for k in self.original_sentences.keys()]

    def next_batch(self,batch_size):

        if batch_size <= 0 or batch_size> self.num_examples:
            raise ValueError("invalid argument for batch size:  {}".format(batch_size))

        if self._index_in_epoch  == 0:
            np.random.shuffle(self.__ids)  # shuffle index

        start = self._index_in_epoch
        # go to the next batch
        if start + batch_size > self.num_examples:
            batch_size = self.num_examples - start

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        indices = self.__ids[start : end]
        batch = {k: (self.processed_sentences[k], self.get_samples_by_sentence_id(k)) for k in indices}

        if end == self.num_examples:
            self.epochs_completed +=1
            self._index_in_epoch = 0


        return batch, self._index_in_epoch == 0


class SupervisedParsing:

    def __init__(self, path):
        self.__ids = []
        self.examples = []
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.num_examples = 0
        self.get_supervised_data(path)

    def get_supervised_data(self, path):
        sents = pickle.load(open(path,'rb'))
        self.num_examples = len(sents)
        self.__ids = [x for x in range(len(sents))]
        self.examples = sents

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        if start == 0:
            np.random.shuffle(self.__ids)  # shuffle index

        # go to the next batch
        elif start + batch_size > self.num_examples:
            self.epochs_completed += 1
            self._index_in_epoch=0
            return  self.next_batch(batch_size)


        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        indices = self.__ids[start : end]
        return [(self.examples[k][0], self.examples[k][1]) for k in indices]


