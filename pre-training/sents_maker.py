from copy import deepcopy
import pickle
from logical_forms_new import *
import  random
import os
import definitions
from seq2seqModel.logical_forms_generator import load_functions
from seq2seqModel.utils import execute
from handle_data import *
from preprocessing import *


LOGICAL_TOKENS_MAPPING_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping')
PARSED_FORMS_PATH = 'temp_sents.txt'

colors = ['yellow', 'blue', 'black']
locs = [('top', 'top'), ('bottom', 'bottom'), ('base', 'bottom')]
ints = ['2', '3', '4', '5', '6', '7']
shapes = ['triangle', 'circle', 'square', ]
quants = [('exactly', 'equal_int'), ('at least', 'ge'), ('at most', 'le')]
ones = ['1']

replacements_dic = {'T_SHAPE' : [('square', ['square']),('triangle', ['triangle']),('circle', ['circle'])],
             'T_COLOR' : [('yellow', ['yellow']),('blue', ['blue']),('black', ['black'])],
             'T_LOC' :  [('top', ['top']),('bottom', ['bottom'])],
             'T_ONE' : [('1', ['1', 'one', 'a'])],
             'T_INT' : [(str(i), [str(i)]) for i in range (2,8)],
             'T_QUANTITY_COMPARE' : [('equal_int', ['exactly']),('le', ['at least']),('ge', ['at most']),
                                     ('lt', ['more than']),('gt', ['less than'])]
             }

logical_tokens_mapping = load_functions(LOGICAL_TOKENS_MAPPING_PATH)

formalization_file = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'formalized words.txt')
formalization_file_2 = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'more_replacements.txt')

def get_sentences_formalized(sentences):
    dict = load_dict_from_txt(formalization_file)
    for i in range(2,10):
        dict[str(i)] = 'T_INT'
    dict["1"] = 'T_ONE'
    dict["one"] = 'T_ONE'
    formalized_sentences =  replace_words_by_dictionary(sentences, dict)
    return formalized_sentences


def print_unique_sents_with_counts(sentences):
    unique = {}
    for s in sentences.values():
        increment_count(unique, s)
    for s, count in sorted(unique.items(), key= lambda kvp : kvp[1], reverse=True):
        print(s,count)



def load_forms(path):
    result = {}
    with open(path) as forms_file:
        for line in forms_file:
            if line.startswith('@'):
                engsent = line[2:].rstrip()
                engsent , form_count = engsent.split('$')
                engsent = engsent.strip()
                form_count = int(form_count.strip())
                result[engsent] = (form_count, [])
            elif line.startswith('~'):
                logsent = line[2:].strip()
                result[engsent][1].append(logsent)
            else:
                continue
    return result

def generate_eng_log_pairs(engsent, logsent, n_pairs):
    eng_words = engsent.split()
    forms = set([w for w in eng_words if w == w.upper() and not w.isdigit()])
    result = []
    for _ in range(n_pairs):
        eng_words = engsent.split()
        log_tokens = logsent.split()
        current_replacements = {}
        for f in forms:
            f_ = f[:-2] if f[-1].isdigit() else f
            current_replacements[f] = random.choice(replacements_dic[f_])
        for i, word in enumerate(eng_words):
            if word in current_replacements:
                logtoken, real_words = current_replacements[word]
                eng_words[i] = random.choice(real_words)

        for form, (logtoken, _) in current_replacements.items():
            for i, tok in enumerate(log_tokens):
                if form in tok:
                    newtok = str.upper(logtoken) if '.' in tok else logtoken
                    log_tokens[i] = tok.replace(form, newtok)

        result.append((" ".join(eng_words), " ".join(log_tokens)))
    return result


def check_generated_forms(forms_doctionary, samples):

    for engsent, (form_count, logsents) in sorted(forms_doctionary.items(), key = lambda k : - k[1][0]):
        for logsent in logsents:
            curr_samples = random.sample(samples, 5)
            generated_forms = generate_eng_log_pairs(engsent, logsent, 5)
            for gen_sent, gen_log in generated_forms:
                for sample in curr_samples:
                    r = execute(gen_log.split(), sample.structured_rep, logical_tokens_mapping)
                    if r is None:
                        print("not compiled:")
                        print(gen_log)
                        print("original=" + logsent)
                        print()

def generate_pairs(forms_doctionary):
    all_pairs =[]
    parsing_dict = {}
    for engsent, (form_count, logsents) in sorted(forms_doctionary.items(), key=lambda k: - k[1][0]):
        for logsent in logsents:
            num = 10 *form_count // len(logsents)
            all_pairs.extend(generate_eng_log_pairs(engsent, logsent, num))


    for k,v in all_pairs:
        parsing_dict[k] = v

    pairs = [(k,v) for k,v in parsing_dict.items()]

    return pairs


def extract_all_sentences_in_given_patterns(sentences, patterns):
    formalized = get_sentences_formalized(sentences)
    result = {}
    for k, s in formalized.items():
        if s in patterns:
            result[k] = sentences[k]
    return result


def sents_maker(path = r'temp_sents.txt'):

    fsents = open(path)



    sents = []
    for line in fsents:
        if line.startswith('@'):
            engsent = line[2:].rstrip()
        elif line.startswith('~'):
            logsent = line[2:].rstrip()
            sents.append([engsent, logsent])
        else:
            continue

    print('beginning with ', len(sents), 'sentences')

    # newsents = sents
    oldsents = []
    # while newsents != []:
    while oldsents != sents:
        oldsents = deepcopy(sents)
        for sent in sents:
            engsent = sent[0]
            logsent = sent[1]
            engwords = engsent.split()
            for i, word in enumerate(engwords):
                if word == word.upper() and word != '1' and word not in ints:
                    sents.remove(sent)
                    if 'COLOR' in word:
                        for color in colors:
                            newlog = logsent.replace(word, color)
                            neweng = engwords
                            neweng[i] = color
                            neweng = ' '.join(neweng)
                            sents.append([neweng, newlog])
                    elif 'SHAPE' in word:
                        for shape in shapes:
                            newlog = logsent.replace(word, shape)
                            neweng = engwords
                            neweng[i] = shape
                            neweng = ' '.join(neweng)
                            sents.append([neweng, newlog])
                    elif 'INT' in word:
                        for inty in ints:
                            newlog = logsent.replace(word, inty)
                            neweng = engwords
                            neweng[i] = inty
                            neweng = ' '.join(neweng)
                            sents.append([neweng, newlog])
                    elif 'LOC' in word:
                        for loc in locs:
                            newlog = logsent.replace(word, loc[1])
                            neweng = engwords
                            neweng[i] = loc[0]
                            neweng = ' '.join(neweng)
                            sents.append([neweng, newlog])
                    elif 'QUANTITY' in word:
                        for quant in quants:
                            newlog = logsent.replace(word, quant[1])
                            neweng = engwords
                            neweng[i] = quant[0]
                            neweng = ' '.join(neweng)
                            sents.append([neweng, newlog])
                    # elif 'ONE' in word:
                    #     for one in ones:
                    #         newlog = logsent
                    #         neweng = engwords
                    #         neweng[i] = one
                    #         neweng = ' '.join(neweng)
                    #         sents.append([neweng, newlog])
                    break

        print('so far ', len(sents), 'sentences')
    print('done! ', len(sents), 'sentences')
    return sents






if __name__ == '__main__':
    pass
    # parsed_forms = load_forms(PARSED_FORMS_PATH)
    # file = open('sents_for_pretain_2', 'rb')
    # pairs = pickle.load(file)
    # file.close()
    # n = len(pairs)
    # np.random.shuffle(pairs)
    # pairs_train = pairs[: int( 0.75 * n)]
    # pairs_validation = pairs[int( 0.75 * n): ]
    # pickle.dump(pairs_train, open('pairs_train', 'wb') )
    # pickle.dump(pairs_validation, open('pairs_validation', 'wb'))
    #
    # print("")
