import json
import numpy as np
import pickle
import definitions
from structured_rep import *
from logical_forms import TokenTypes
from sentence_processing import preprocess_sentences, replace_rare_words_with_unk
from seq2seqModel.hyper_params import SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH
import sys

np.random.seed(1)

class DataSet(Enum):
    TRAIN = 'train',
    DEV = 'dev',
    TEST = 'test',
    TEST2 = 'hidden_test'

if len(sys.argv) == 2:
    paths = {DataSet.TRAIN: definitions.TRAIN_JSON,
             DataSet.DEV: definitions.DEV_JSON,
             DataSet.TEST: definitions.TEST_JSON,
             DataSet.TEST2: definitions.TEST2_JSON}
else:
    paths = {DataSet.TRAIN: definitions.TRAIN_JSON,
             DataSet.DEV: definitions.DEV_JSON,
             DataSet.TEST: definitions.TEST_JSON}


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
            line = json.dump(sample, output_file)
            output_file.write('\n')
    return


def build_data(data, preprocessing_type=None, use_unk=True):
    '''

    :param data: a deserialized version of a dataset json file: List[List[List[Dict[str: str]]]]
    :param preprocessing_type: the type of setnece preprocessing to be used
    :param use)unk : whether to replace rare word words <UNK> tokens
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
    if preprocessing_type == 'abstraction':
        dicts_dict = {idx: sent[1] for idx, sent in sentences.items()}
        sentences = {idx: sent[0] for idx, sent in sentences.items()}
    if use_unk:
        sentences = replace_rare_words_with_unk(sentences)

    for s in samples:
        s_index = int(str.split(s.identifier, "-")[0])
        s.sentence = sentences[s_index]
        if preprocessing_type == 'abstraction':
            s.abstraction_dict = dicts_dict[s_index]

    return samples, sentences


class CNLVRDataSet:
    """
    A wrapper class for an instance of a data set from CNLVR (i.e. train, dev, or test).
    This class encapsulates all the processing done for loading the data, and provides
    functionality for going over the data set one batch after another, as well as some other
    options.
    """

    def __init__(self, dataset):
        self.__dataset = dataset
        self.original_sentences = {}
        self.processed_sentences = {}
        self.samples = {}
        self.__ids = []
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.__ids_by_complexity = []
        self.pictures = {}
        self.sentences_quardpled_ids = []
        self.processed_sentences_singles = {}
        self.get_data(paths[dataset])


    @property
    def name(self):  # i.e. 'TRAIN'
        return self.__dataset.name

    @property
    def num_examples(self):
        return len(self.__ids)

    @property
    def num_single_examples(self):
        return len(self.sentences_quardpled_ids)

    def build_single_lists(self):
        # created new ids for treating the sentences and pictures in 1:1 ratio
        sents_num = len(self.original_sentences.keys())
        k=0
        for i in range(sents_num):
            for j in range(4):

                sample_original_name="{0}-{1}".format(list(self.original_sentences.keys())[i], j)
                sample_new_name = "{0}".format(k)
                if sample_original_name in self.samples:
                    self.pictures[sample_new_name] = self.samples[sample_original_name]
                    self.sentences_quardpled_ids.append(k)
                    self.processed_sentences_singles[k] = self.processed_sentences[list(self.original_sentences.keys())[i]]
                    k += 1

    def get_samples_by_sentence_id(self, sentence_id):
        samples_ids = ["{0}-{1}".format(sentence_id, i) for i in range(4)]
        return [self.samples[sample_id] for sample_id in samples_ids if sample_id in self.samples]

    def get_single_sample_for_sentence(self,sentence_id_q):
        return [self.pictures["{0}".format(sentence_id_q)]]

    def get_sentence_by_id(self, sentence_id, original=False):
        if original:
            return self.original_sentences[sentence_id]
        return self.processed_sentences[sentence_id]

    def get_data(self, path):
        # this methods handles all loading and processing of the data set and thus is called
        # at initialization.

        data = read_data(path)
        sentences = {}
        for line in data:
            self.samples[line["identifier"]] = Sample(line)
            s_index = int(str.split(line["identifier"], "-")[0])
            if s_index not in sentences:
                sentences[s_index] = line["sentence"]

        self.original_sentences = preprocess_sentences(sentences, processing_type='shallow')

        if self.__dataset == DataSet.TRAIN:
            mode = None
            counts_file = None
        else:
            mode = 'r'
            counts_file = definitions.TOKEN_COUNTS_PROCESSED

        if definitions.ABSTRACTION:
            self.processed_sentences = preprocess_sentences(sentences, mode=mode, processing_type='abstraction')
            dicts_dict = {idx: sent[1] for idx, sent in self.processed_sentences.items()}
            self.processed_sentences = {idx: sent[0] for idx, sent in self.processed_sentences.items()}
        else:
            self.processed_sentences = preprocess_sentences(sentences, mode=mode, processing_type='deep')
        self.processed_sentences = replace_rare_words_with_unk(self.processed_sentences, counts_file)

        # if self.__dataset == DataSet.TRAIN:
        #     self.processed_sentences = \
        #         preprocess_sentences(sentences, mode=None, processing_type='deep')
        #
        # else:
        #     self.processed_sentences = \
        #         preprocess_sentences(sentences, mode='r', processing_type='deep')
        #
        # if self.__dataset == DataSet.TRAIN:
        #     self.processed_sentences = replace_rare_words_with_unk(self.processed_sentences)
        #
        # else:
        #     self.processed_sentences = \
        #         replace_rare_words_with_unk(self.processed_sentences,
        #                                     definitions.TOKEN_COUNTS_PROCESSED)


        for s in self.samples.values():
            s_index = int(str.split(s.identifier, "-")[0])
            s.sentence = self.processed_sentences[s_index]
            if definitions.ABSTRACTION:
                s.abstraction_dict = dicts_dict[s_index]

        self.__ids = [k for k in self.original_sentences.keys()]
        #self.sentences_quardpled_ids = []
        self.build_single_lists()

    def use_subset_by_sentnce_condition(self, f_s):
        """
        limits the dataset to sentences that follow some condition only.
        :param f_s:  a boolean function on ids
        """

        new_ids = []
        for k, s in self.processed_sentences.items():
            if f_s(s):
                new_ids.append(k)
        self.__ids = new_ids

    def use_subset_by_images_condition(self, f_im):
        """
        limits the dataset to sentences whose related imaes follow some rule
        :param f_s:  f_im is a boolean function on a set of samples
        """

        new_ids = []
        for k, s in self.processed_sentences.items():
            related_samples = self.get_samples_by_sentence_id(k)
            if f_im(related_samples):
                new_ids.append(k)
        self.__ids = new_ids

    def ignore_all_true_samples(self):
        """

        limits the dataset to sentences that are not true about all their images -
        should help avoid spurious signal (there are about 10% such sentences in the training set)
        """
        all_true_filter = lambda s_samples: not all([s.label == True for s in s_samples])
        self.use_subset_by_images_condition(all_true_filter)

    def sort_sentences_by_complexity(self, complexity_measure, n_classes):
        '''
        sorts the data into n_classes sets by some measure of complexity of the sentences.
         can be used for curriculum learning
        '''
        self.__ids_by_complexity = []
        ids_sorted_by_sentence_length = sorted(self.processed_sentences.keys(), key=
        lambda key: complexity_measure(self.processed_sentences[key]))
        class_size = len(self.processed_sentences) // n_classes
        for i in range(n_classes):
            self.__ids_by_complexity.append(ids_sorted_by_sentence_length[
                                            class_size * i: min(class_size * i + class_size,
                                                                len(self.processed_sentences))])
        return

    def choose_levels_for_curriculum_learning(self, levels):
        self.__ids = [idx for idx in set(ind for level in levels for ind in self.__ids_by_complexity[level])]

    def restart(self):
        '''
        restart the state of the data set (the is no need to reload it from disk - just call this method)
        '''
        self.__ids = [k for k in self.original_sentences.keys()]
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.sentences_quardpled_ids=[x for x in self.processed_sentences_singles.keys()]

    def next_batch(self, batch_size):
        '''

        return the next batch of  (sentence, related samples) pairs.
        also habdles the logic of moving between epochs.
        '''

        if batch_size <= 0 or batch_size > self.num_examples:
            raise ValueError("invalid argument for batch size:  {}".format(batch_size))

        if self._index_in_epoch == 0:
            np.random.shuffle(self.__ids)  # shuffle index

        start = self._index_in_epoch
        # go to the next batch
        if start + batch_size > self.num_examples:
            batch_size = self.num_examples - start

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        indices = self.__ids[start: end]
        batch = {k: (self.processed_sentences[k], self.get_samples_by_sentence_id(k)) for k in indices}

        if end == self.num_examples:
            self.epochs_completed += 1
            self._index_in_epoch = 0

        return batch, self._index_in_epoch == 0

    def next_batch_singles(self, batch_size):
        '''

        return the next batch of  (sentence, related samples) pairs.
        also habdles the logic of moving between epochs.
        '''

        if batch_size <= 0 or batch_size > self.num_single_examples:
            raise ValueError("invalid argument for batch size:  {}".format(batch_size))

        if self._index_in_epoch == 0:
            np.random.shuffle(self.sentences_quardpled_ids)  # shuffle index

        start = self._index_in_epoch
        # go to the next batch
        if start + batch_size > self.num_single_examples:
            batch_size = self.num_single_examples - start

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        indices = self.sentences_quardpled_ids[start: end]
        batch = {k: (self.processed_sentences_singles[k], self.get_single_sample_for_sentence(k)) for k in indices}

        if end == self.num_single_examples:
            self.epochs_completed += 1
            self._index_in_epoch = 0

        return batch, self._index_in_epoch == 0


class DataSetForSupervised:
    """
    a simpler version of the class above, used only for the supervised learning
    """

    def __init__(self, path):
        self.__ids = []
        self.examples = []
        self._index_in_epoch = 0
        self.epochs_completed = 0
        self.num_examples = 0
        self.get_supervised_data(path)

    def get_supervised_data(self, path):
        sents = pickle.load(open(path, 'rb'))
        self.num_examples = len(sents)
        self.__ids = [x for x in range(len(sents))]
        self.examples = sents

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        if start == 0:
            np.random.shuffle(self.__ids)  # shuffle index

        # go to the next batch
        elif start + batch_size > self.num_examples:
            self.epochs_completed += 1
            self._index_in_epoch = 0
            return self.next_batch(batch_size)

        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        indices = self.__ids[start: end]
        return [(self.examples[k][0], self.examples[k][1]) for k in indices]


def load_functions(filename):
    """
    loads from file a dictionary of all valid tokens in the formal language we use.
    each token is is defines by its name, its return types, and its argument types.
    tokens that represent known entities, like ALL_BOXES, or Color.BLUE are treated as
    functions that take no arguments, and their return type is their own type, i.e.
    set<set<Item>>, and set<Color>, rspectively.
    """
    functions_dict = {}
    with open(filename) as functions_file:
        for i, line in enumerate(functions_file):
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith('#'):
                continue
            entry = line.split()

            split_idx = entry.index(':') if ':' in entry else len(entry)
            entry, necessary_words = entry[:split_idx], entry[split_idx:]

            if len(entry) < 3 or not entry[1].isdigit() or int(entry[1]) != len(entry) - 3:
                print("could not parse function in line  {0}: {1}".format(i, line))
                # should use Warning instead
                continue
            token, return_type, args_types = entry[0], entry[-1], entry[2:-1]
            functions_dict[token] = TokenTypes(return_type=return_type, args_types=args_types,
                                               necessity=necessary_words)
        if SENTENCE_DRIVEN_CONSTRAINTS_ON_BEAM_SEARCH:
            functions_dict['1'] = TokenTypes(return_type='int', args_types=[], necessity=['1', 'one', 'a'])
            functions_dict.update(
                {str(i): TokenTypes(return_type='int', args_types=[], necessity=[str(i)]) for i in range(2, 10)})
        else:
            functions_dict['1'] = TokenTypes(return_type='int', args_types=[], necessity=[])
            functions_dict.update(
                {str(i): TokenTypes(return_type='int', args_types=[], necessity=[]) for i in range(2, 10)})

    return functions_dict
