import unittest
from sandbox.logical_forms_new import *
import definitions
import display_images
from handle_data import *
from preprocessing import clean_sentence


ALPHABET = "abcdefghijklmnopqrstuvwxyz"
PARSING_EXAMPLES_PATH = os.path.join(definitions.DATA_DIR, "parses for check")
PARSING_EXAMPLES_TOKENS_ONLY_PATH = os.path.join(definitions.DATA_DIR, "parses for check as tokens")
TOKENS_INVENTORY = os.path.join(definitions.DATA_DIR, "token mapping")
KNOWN_MISTAKES = os.path.join(definitions.DATA_DIR, "wrong labels ids")

TOKENS_ONLY = True # try both True and false




def load_parsed_examples(path):
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

tokens_dic = load_token_inventory(TOKENS_INVENTORY)
parsed_samples = load_parsed_examples(PARSING_EXAMPLES_PATH) if not TOKENS_ONLY else \
                                        load_parsed_examples(PARSING_EXAMPLES_TOKENS_ONLY_PATH)
known_mistakes = load_known_mistakes(KNOWN_MISTAKES)


def test_parsing_on_samples(sentence, samples, only_tokens = TOKENS_ONLY, show_mistakes = 'all', show_erros = True):

    n_errors = 0
    n_mistakes = 0

    logical_forms = parse(sentence, parsed_samples)
    n_logical_forms= len(logical_forms)
    for form in logical_forms:
        for sample in samples:
            if only_tokens:
                form = process_token_sequence(form.split(), tokens_dic)
            try:
                result = run_logical_form(form, sample.structured_rep)

            except (TypeError, SyntaxError, ValueError, AttributeError,
                    RuntimeError, RecursionError, Exception, NotImplementedError) as err:
                n_errors+=1
                if show_erros:
                    print("an error accured while executing logical form:")
                    print(form)
                    print(err)
                    input("press any key to continue")
                    break

            if result != sample.label:
                n_mistakes+=1
                if show_mistakes=='all' or show_mistakes=='ignore_known_mistakes' \
                     and sample.identifier not in known_mistakes:

                    print("prediction doesn't match label on sample {}:".format(sample.identifier))
                    print("sentence: {}".format(sample.sentence))
                    print("logical form: {}".format(form))
                    print("predicted label: {}".format(result))
                    print("actual label: {}".format(sample.label))
                    print(sample.structured_rep.boxes)
                    show_image = input("show image (y/n)?\n") == 'y'
                    if show_image:
                        display_images.show_image(sample.identifier)
                    input("press any key to continue")

    return n_logical_forms, n_errors, n_mistakes


def test_parsed_forms():
    data, sentences = build_data(read_data(TRAIN_JSON), preprocessing_type='shallow')
    has_parsing = [k for k,s in sentences.items() if s in parsed_samples]

    for k in has_parsing:
        s = (sentences[k])
        samples = [sample for sample in data if sample.sentence == s]
        test_parsing_on_samples(s, samples, show_mistakes='ignore_known_mistakes')
    return



class TestLogicalForms(unittest.TestCase):

    def test_something(self):
        pass

    def test_true(self):
        self.assertTrue(5>2)




if __name__ == '__main__':
    test_parsed_forms()
    #unittest.main()
