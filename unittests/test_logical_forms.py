"""
testing the logical forms. right now, does not include real unit tests for the logical forms functions,
but rather enables checking pre-parsed sentences (from our examples) on their corresponding structured representations.
This also enables to find bugs or problems in the way some logical forms are defined.Note that there are many parameters
to play with here, and this can help us also when evaluating parses generated from our future learning
algorithm in-sha-la.
"""

import unittest
import os
import definitions
import display_images
from handle_data import *
from logical_forms import *
from preprocessing import clean_sentence
from seq2seqModel.logical_forms_generation import load_functions

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
PARSING_EXAMPLES_PATH = os.path.join(definitions.DATA_DIR, "parsed sentences", "parses for check")
PARSING_EXAMPLES_TOKENS_ONLY_PATH = os.path.join(definitions.DATA_DIR, "parsed sentences", "parses for check as tokens")
TOKENS_INVENTORY = os.path.join(definitions.DATA_DIR, "logical forms","token mapping")
KNOWN_MISTAKES = os.path.join(definitions.ROOT_DIR,'unittests', "wrong labels ids")
INCORRECT_PARSES = os.path.join(definitions.DATA_DIR, 'incorrect_first_in_beam')

TOKENS_ONLY = True # try both True and false



### loading methods

def load_parsed_examples(path):
    """
    
    :param path: a file with examples of sentences and their suggested parsing(s) 
    :return: a dictionary mapping english sentences to their parsing(s)
    """
    parsing_dic ={}
    sentence = None
    with open(path) as parsed_examples:
        for line in parsed_examples:
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith('#'):
                continue
            if line[0].isdigit():
                line = line.lower()
                start = min(line.find(ch) for ch in ALPHABET if line.find(ch)>0)
                sentence_number = clean_sentence(line[: start])
                sentence = clean_sentence(line[start :])
                parsing_dic[sentence] = (sentence_number,[])
            else:
                if sentence is None:
                    break
                parsing_dic[sentence][1].append(line)
    return parsing_dic


def load_token_inventory(path):
    # in order to process parses that are in 'token-only' form, we need to
    # know hoe many arguments each token takes.
    tokens_dic ={}
    with open(path) as tokens:
        for line in tokens:
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith('#'):
                continue
            l = line.split()
            tokens_dic[l[0]] = int(l[1])
    return tokens_dic


def load_known_mistakes(path):
    """
    This methods loads the ids of samples with wrong label from a designated file.
    Unfortunately, the data is full of mistaken or 'controversial' labels.
    In some cases the true/false label attached to a sample is just wrong. In other cases,
     there are problems with anbigious sentences and definitions. Some examples are strong cardinality 
     vs. soft cardinality (equal or greater-equal), whether to interpret 'above' as 'the one above' or
     'all objects above', whether a single-block tower is considered a tower etc.
     anyway, it is good idea to collect the ids of such samples. Then you can ask to ignore
    these samples in test_parsing_on_samples.
    
     
    """
    known_mistakes = set()
    with open(path) as tokens:
        for line in tokens:
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith('#'):
                continue
            known_mistakes.add(str.rstrip(line[:6]))
    return known_mistakes


#load needed data

tokens_dic = load_functions(TOKENS_INVENTORY)
parsed_samples = load_parsed_examples(PARSING_EXAMPLES_PATH) if not TOKENS_ONLY else \
                                        load_parsed_examples(PARSING_EXAMPLES_TOKENS_ONLY_PATH)
known_mistakes = load_known_mistakes(KNOWN_MISTAKES)


def test_parsing_on_samples(sentence, samples, only_tokens=TOKENS_ONLY, show_mistakes='all', show_erros=True,
                            parsing_dic = parsed_samples):
    '''

    :param sentence: the sentence whose parsing you want to check 
    :param samples: samples related to this sentence
    :param only_tokens: when using pre-parsed forms, whether to use the token-only form or the final form
     (with brackets and commas, that python can just execute using eval)
    :param show_mistakes: whether to report discrepancy between the predicted and actual label.
    'all' - shows all such discrepancies
    'ignore_known_mistakes' - show only mistakes that are not in the known mistakes list
    otherwise don't report mistakes
    :param show_erros: whether to report the erros in real time
    :return: 
    '''
    n_errors = 0
    n_mistakes = 0

    logical_forms = parse(sentence, parsing_dic)
    n_logical_forms = len(logical_forms)

    for form in logical_forms:
        next_sentence = False
        if only_tokens:
            form = process_token_sequence(form, tokens_dic)
        for sample in samples:
            if next_sentence:
                continue

            try:
                result = run_logical_form(form, sample.structured_rep)

            except (TypeError, SyntaxError, ValueError, AttributeError,
                    RuntimeError, RecursionError, Exception, NotImplementedError) as err:
                n_errors += 1
                if show_erros:
                    print("an error accured while executing logical form:")
                    print(form)
                    print(err)
                    input("press any key to continue")
                    break

            if result != sample.label:
                n_mistakes += 1
                if show_mistakes == 'all' or show_mistakes == 'ignore_known_mistakes' \
                        and sample.identifier not in known_mistakes:

                    print("prediction doesn't match label on sample {}:".format(sample.identifier))
                    print("sentence: {}".format(sample.sentence))
                    print("logical form: {}".format(form))
                    print("predicted label: {}".format(result))
                    print("actual label: {}".format(sample.label))
                    print(sample.structured_rep.boxes)
                    next_sentence = input("next sentence? (y/n)?\n") == 'y'
                    if next_sentence:
                        return True

                    show_image = input("show image (y/n)?\n") == 'y'
                    if show_image:
                        display_images.show_image(sample.identifier)
                    input("press any key to continue")

    return False

def test_forms_from_beam():
    file = open(INCORRECT_PARSES, 'rb')
    d = pickle.load(file)
    train = CNLVRDataSet(definitions.TRAIN_JSON)
    parsing_dict = {s: (id_s, [prog.token_seq]) for id_s, (s, prog) in d.items()}
    known_parsing_mistakes = open('known_parse_errors2.txt', 'w')
    for id_s, (s, prog) in d.items():
        if 'touch' in s:
            samples = train.get_samples_by_sentence_id(id_s)
            if test_parsing_on_samples(s, samples, only_tokens=TOKENS_ONLY, show_mistakes='all',  parsing_dic = parsing_dict):
                known_parsing_mistakes.write(s + '\n')


    known_parsing_mistakes.close()

def test_almost_touching():
    train = CNLVRDataSet(definitions.TRAIN_JSON)
    while True:
        batch = train.next_batch(100)
        ids = [key for key in batch.keys()]
        sentences, samples = zip(*[batch[k] for k in ids])
        for i in range(len(sentences)):
            if 'closely' in sentences[i]:
                print(sentences[i])
                samples = train.get_samples_by_sentence_id(ids[i])
                for sample in samples:
                    print(sample.label)
                    if input("show image? (y/n)?\n") == 'y':
                        display_images.show_image(sample.identifier)


class TestLogicalForms(unittest.TestCase):

    def test_something(self):
        pass

    def test_true(self):
        self.assertTrue(5>2)




if __name__ == '__main__':
    test_almost_touching()
    #unittest.main()
