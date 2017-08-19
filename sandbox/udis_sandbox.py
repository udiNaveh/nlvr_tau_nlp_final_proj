import string
import nltk



from handle_data import *
from general_utils import increment_count
import display_images
from preprocessing import load_ngrams

color_words = ["blue", "yellow", "black"]
shape_words = ["circle", "triangle", "square"]
integers = [str(i) for i in range(1,10)]
integer_words = "one two three four five six seven".split()


def formalize_sentence(s):
    s = [w for w in s.split()]

    forms = [(integers, [], 'n'), (shape_words, [], 's'), (color_words, [], 'c')]

    for i,w in enumerate(s):
        # remove plural and convert integer words:
        if w.endswith('s') and w[:-1] in shape_words:
            s[i] = w[:-1]
        if w in integer_words:
            s[i] = integers[integer_words.index(s[i])]
        #formalize
        for f in forms:
            if s[i] in f[0]:
                if s[i] not in f[1]:
                    f[1].append(s[i])
                s[i] = f[2] + str(f[1].index(s[i]))
    return s

def find_rare_words():
    data, sentences = build_data(read_data(TRAIN_JSON), preprocess= True)
    token_counts = load_ngrams(TOKEN_COUNTS, 1)
    for kvp in sorted(token_counts.items(), key= lambda kvp : kvp[1]):
        print(kvp)
    sentence_with_rare_words = {}
    for k, s in sentences.items():
        tokens = s.split()
        rare_tokens = [(t, token_counts.get(t, 0)) for t in tokens if token_counts.get(t, 0) <=5]
        if len(rare_tokens) > 0:
            sentence_with_rare_words[k] = rare_tokens

    for k, v in sentence_with_rare_words.items():
        print (sentences[k], v)






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






if __name__ == '__main__':
    #build_data(read_data(TRAIN_JSON), preprocess=True)
    call_api('position')
