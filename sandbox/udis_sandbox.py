import string
import nltk
import tensorflow as tf
from handle_data import build_data

from handle_data import *
from general_utils import increment_count
import display_images
from preprocessing import *

color_words = ["blue", "yellow", "black"]
color_tokens = []
shape_words = ["circle", "triangle", "square"]
integers = [str(i) for i in range(1,10)]
integer_words = "one two three four five six seven".split()
location_words = ["top", "bottom", "base", "first", "second", "third"]
quantity_words = ["at most", "at least", "more than", "less than", "exactly"]
size_words = ["medium", "small", "big"]


TOKEN_REP_PATH = r'C:\Users\ASUS\Dropbox\אודי\לימודים\שנה ג סמסטר ב\NLP\project\CNLVR\tokens important'
MORE_REP_PATH = r'C:\Users\ASUS\Dropbox\אודי\לימודים\שנה ג סמסטר ב\NLP\project\CNLVR\more_replacements'


def read_dict_from_txt(path):
    """
    loads a dictionary from a text file in which every line is in the format
    key : value
    assumes that keys are unique
    :return: a dictionary mapping string to strings
    """

    reps = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line)==0 or line.startswith('#'):
                continue
            kvp = line.split(':')
            if len(kvp)!=2:
               continue
            key = line[0].strip()
            value = line[1].strip()
            reps[key] = value
    return reps


more_rep_dict = get_more_rep_dict()

def load_token_replacement(path):
    count_dic = {}
    sets = []
    with open(path) as tokens_file:
        for line in tokens_file:
            found =False
            line = line.split()
            token = line[0]
            count = int(line[1])
            count_dic[token] = count
            new_set = set()
            new_set.add(token)
            if len(line)>2:
                alternatives = line[2:]
                new_set.update(alternatives)
            for i, s in enumerate(sets):
                if len(s.intersection(new_set))>0:
                    sets[i] = s.union(new_set)
                    found = True
                    break
            if not found:
                sets.append((new_set))
    return sets, count_dic


def get_replacement_dict(path):
    sets, count_dict = load_token_replacement(path)
    replacement_dict= {}
    for s in sets:
        if len(s)> 1:
            replacement = max([t for t in s if '-' not in t and t!='block'], key = lambda token: count_dict.get(token, 0))
            for token in s:
                if token!=replacement:
                    replacement_dict[str.replace(token, '-', ' ')] = replacement
    return replacement_dict


def get_sentences_with_replacements():
    data = read_data(train)
    sentences = {}
    for line in data:
        s_index = int(str.split(line["identifier"], "-")[0])
        if s_index not in sentences:
            sentences[s_index] = line["sentence"]

    sentences = preprocess_sentences(sentences, processing_type='lemmatize')
    rep_dict = get_replacement_dict(TOKEN_REP_PATH)
    for k,v in rep_dict.items():
        print ("{0} : {1}".format(k,v))
    for k in sentences.keys():
        for t in rep_dict.keys():
            sentences[k] = (sentences[k] + ' ').replace(' {} '.format(t), ' {} '.format(rep_dict[t]))
    return sentences

def get_sentences_with_unks():
    sentences = get_sentences_with_replacements()
    sentences_tokenized = {k: s.split() for k, s in sentences.items()}
    unigrams = get_ngrams_counts(sentences_tokenized.values(), 1)[0]
    for k, s in sentences_tokenized.items():
        sentences_tokenized[k] = " ".join([w  if unigrams.get(w,0)>=5 else '<UNK>' for w in s])
    return sentences_tokenized

def get_sentences_formalized():
    sentences = get_sentences_with_unks()
    formalized_sentences = {k : formalize_sentence(s) for k,s in sentences.items()}
    count_dict = {}
    for s in formalized_sentences.values():
        increment_count(count_dict, s)


    ordered = sorted(count_dict.items(), key = lambda kvp : kvp[1], reverse=True)
    return ordered





#
# def formalize_sentence(s):
#     s = [w for w in s.split()]
#
#     forms = [(integers, [], 'n'), (shape_words, [], 's'), (color_words, [], 'c')]
#
#     for i,w in enumerate(s):
#         # remove plural and convert integer words:
#         if w.endswith('s') and w[:-1] in shape_words:
#             s[i] = w[:-1]
#         if w in integer_words:
#             s[i] = integers[integer_words.index(s[i])]
#         #formalize
#         for f in forms:
#             if s[i] in f[0]:
#                 if s[i] not in f[1]:
#                     f[1].append(s[i])
#                 s[i] = f[2] + str(f[1].index(s[i]))
#     return s


def formalize_sentence(s):
    if isinstance(s, str):
        s = [w for w in s.split()]

    forms = [(integers, [], 'n'), (shape_words, [], 's'), (color_words, [], 'c')]

    for i,w in enumerate(s):
        if w in integers or w in integer_words:
            s[i] = 'T_INT'
        if w in color_words:
            s[i] = 'T_COLOR'
        if w in shape_words:
            s[i] = 'T_SHAPE'
        if w in size_words:
            s[i] = 'T_SIZE'
        if w in location_words:
            s[i] = 'T_LOC'

    s = " ".join(s)
    for q in quantity_words:
        if q in s:
            s= s.replace(q, 'T_quantity')
    s = " " + s + " "
    for pair in more_rep_dict:
        s = s.replace(' ' +pair[0] + ' ', ' '+pair[1] + ' ')

    return s[1:-1]

# def find_rare_words():
#     data, sentences = build_data(read_data(TRAIN_JSON), preprocess= True)
#     token_counts = load_ngrams(TOKEN_COUNTS, 1)
#     for kvp in sorted(token_counts.items(), key= lambda kvp : kvp[1]):
#         print(kvp)
#     sentence_with_rare_words = {}
#     for k, s in sentences.items():
#         tokens = s.split()
#         rare_tokens = [(t, token_counts.get(t, 0)) for t in tokens if token_counts.get(t, 0) <=5]
#         if len(rare_tokens) > 0:
#             sentence_with_rare_words[k] = rare_tokens
#
#     for k, v in sentence_with_rare_words.items():
#         print (sentences[k], v)






def unique_sentences_count(sentences):

    tokenized_clean_sentences = preprocess_sentences(sentences)
    formalized_sentences = {k : formalize_sentence(s) for k,s in tokenized_clean_sentences.items()}

    # for k, s in sentences.items():
    #     # includes illegal word
    #     if len([w for w in s.split(" ") if w.lower() not in vocab and not str.isdigit(w)]) > 0:
    #         print("original    = {}".format(s))
    #         print("corredrcted = {}".format(" ".join(tokenized_clean_sentences[k])))
    #         print("formalized  = {}".format(" ".join(formalized_sentences[k])))



    for sentences_list in [sentences.values(), [str.join(" ", tok) for tok in tokenized_clean_sentences.values()],
                           [str.join(" ", tok) for tok in formalized_sentences.values()]]:

        unique = set(sentences_list)
        sentences_frequency = {}
        for s in sentences_list:
            increment_count(sentences_frequency, s)

        ordered = sorted(sentences_frequency.items(), key=lambda kvp: kvp[1], reverse=True)
        frequency_meta = {}
        for kvp in ordered:
            increment_count(frequency_meta, kvp[1])

        print("out of {0} sentences in the training set, onlt {1} are unique :".format(len(sentences), len(unique)))

        print()
        for item in sorted(frequency_meta.items(), reverse=True):
            print("{0} sentences appear {1} times each".format(item[1], item[0]))

        print("#######")
        print()



    sents_with_relations = [s for s in sentences_list if "above" in s or "below" is s]
    for s in sents_with_relations:
        pass #print(s)
    return


def select_integers(k, min, max):
    if k> max-min or k==0:
        return [set()]
    return [set([min]).union(s) for s in select_integers(k-1, min+1, max)] + [s for s in select_integers(k, min+1, max) if len(s)>0]


def select(_set, k):
    l = list(_set)
    return [[l[i] for i in idx] for idx in select_integers(k, 0, len(l))]


def gen_all_NPitem():
    res = []
    pattren = ['T_SIZE', 'T_COLOR', 'T_SHAPE']
    with open('translations') as f:
        for line in f:
            line.split()



if __name__ == '__main__':
    get_sentences_with_replacements()
    print("")
