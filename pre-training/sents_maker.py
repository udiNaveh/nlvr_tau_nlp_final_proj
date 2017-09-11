from copy import deepcopy
import pickle
from logical_forms import *
import  random
import os
import definitions
from seq2seqModel.logical_forms_generation import *
from seq2seqModel.utils import execute
from data_manager import *
from sentence_processing import *


LOGICAL_TOKENS_MAPPING_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping_limitations')
PARSED_FORMS_PATH = os.path.join(definitions.ROOT_DIR, 'pre-training', 'temp_sents_new')
WORD_EMBEDDINGS_PATH = os.path.join(definitions.ROOT_DIR, 'word2vec', 'embeddings_10iters_12dim')

embeddings_file = open(WORD_EMBEDDINGS_PATH, 'rb')
embeddings_dict = pickle.load(embeddings_file)
embeddings_file.close()


colors = ['yellow', 'blue', 'black']
locs = [('top', 'top'), ('bottom', 'bottom')]
ints = ['2', '3', '4', '5', '6', '7']
shapes = ['triangle', 'circle', 'square', ]
quants = [('exactly', 'equal_int'), ('at least', 'ge'), ('at most', 'le')]
ones = ['1']

replacements_dic = {'T_SHAPE' : [('square', ['square']),('triangle', ['triangle']),('circle', ['circle'])],
             'T_COLOR' : [('yellow', ['yellow']),('blue', ['blue']),('black', ['black'])],
             'T_LOC' :  [('top', ['top']),('bottom', ['bottom'])],
             'T_ONE' : [('1', ['1', 'one'])],
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
    while len(result) <n_pairs:
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

        # check sentence is good:
        eng_sent = " ".join(eng_words)
        log_sent = " ".join(log_tokens)
        if 'gt 1' in log_sent or 'ge 1' in log_sent:
            continue
        bad_int_use = False
        for j in (3,4,5,6,7):
            if np.random.rand() < 0.10 * j:
                continue
            if '{} tower'.format(j) in eng_sent or '{} box'.format(j) in eng_sent\
                    or '{} ALL_BOXES'.format(j) in log_sent:
                bad_int_use =True
                break


        if not bad_int_use:
            result.append((eng_sent, log_sent))
    return result


def check_generated_forms(forms_dictionary, samples):
    next_token_probs_getter = lambda pp: (pp.get_possible_continuations(), [0.1 for p in pp.get_possible_continuations()])


    i = 0
    for engsent, (form_count, logsents) in sorted(forms_dictionary.items(), key = lambda k : - k[1][0]):
        for logsent in logsents:
            curr_samples = random.sample(samples, 1)
            generated_forms = generate_eng_log_pairs(engsent, logsent, 5)
            for gen_sent, gen_log in generated_forms:
                i+=1
                print(i)
                for word in gen_sent.split():
                    if word not in embeddings_dict:
                        print (word)
                for sample in curr_samples:
                    r = execute(gen_log.split(), sample.structured_rep, logical_tokens_mapping)
                    if r is None:
                        print("not compiled:")
                        print(gen_log)
                        print("original=" + logsent)
                        print()
                if not "filter filter filter" in gen_log:
                    try:
                        prog = program_from_token_sequence(next_token_probs_getter, gen_log.split(), logical_tokens_mapping,
                                                    original_sentence=gen_sent)
                    except ValueError:

                        print(gen_sent)
                        print(gen_log)


def generate_pairs(forms_doctionary):
    all_pairs =[]
    parsing_dict = {}
    for engsent, (form_count, logsents) in sorted(forms_doctionary.items(), key=lambda k: - k[1][0]):
        for logsent in logsents:
            num = int(50 *form_count**(0.8)) // len(logsents)
            all_pairs.extend(generate_eng_log_pairs(engsent, logsent, num))


    for k,v in all_pairs:
        parsing_dict[k] = v

    pairs = [(k,v) for k,v in parsing_dict.items()]

    n = len(pairs)
    np.random.shuffle(pairs)
    pairs_train = pairs[: int( 0.75 * n)]
    pairs_validation = pairs[int( 0.75 * n): ]

    return pairs_train, pairs_validation

def generate_pairs_woth_train_val_separation(forms_doctionary):

    pairs_train = []
    pairs_validation = []
    for engsent, (form_count, logsents) in sorted(forms_doctionary.items(), key=lambda k: - k[1][0]):
        if np.random.rand() <0.25:
            pairs = pairs_validation
        else:
            pairs = pairs_train

        for logsent in logsents:
            num = 10 *form_count // len(logsents)
            pairs.extend(generate_eng_log_pairs(engsent, logsent, num))


    return pairs_train, pairs_validation

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
    parsed_forms = load_forms(PARSED_FORMS_PATH)
    samples, sentences = build_data(read_data(TRAIN_JSON), preprocessing_type='shallow')
    #check_generated_forms(parsed_forms, samples)
    train, validation = generate_pairs(parsed_forms)
    # train = open(os.path.join(definitions.DATA_DIR, 'parsed sentences', 'pairs_train3'), 'rb')
    # train = pickle.load(train)
    # validation = open(os.path.join(definitions.DATA_DIR, 'parsed sentences', 'pairs_validation3'), 'rb')
    # validation = pickle.load(validation)
    # train_sep, validation_sep = generate_pairs_woth_train_val_separation(parsed_forms)
    # train_sep = list(set(train_sep))
    # validation_sep = list(set(validation_sep))
    # np.random.shuffle(train_sep)
    # np.random.shuffle(validation_sep)
    # train_sep = train_sep[:len(train)]
    # validation_sep = validation_sep[:len(train)]


    # train_sep = open(os.path.join(definitions.DATA_DIR, 'parsed sentences', 'pairs_train_sep'), 'rb')
    # train_sep = pickle.load(train_sep)
    # validation_sep = open(os.path.join(definitions.DATA_DIR, 'parsed sentences', 'pairs_validation_sep'), 'rb')
    # validation_sep = pickle.load(validation_sep)
    pickle.dump(validation, open(os.path.join(definitions.DATA_DIR, 'parsed sentences', 'pairs_validation3'), 'wb'))
    pickle.dump(train, open(os.path.join(definitions.DATA_DIR, 'parsed sentences', 'pairs_train3'), 'wb'))
    # pickle.dump(validation, open(os.path.join(definitions.DATA_DIR, 'parsed sentences', 'pairs_validation2'), 'wb'))



    print("")
