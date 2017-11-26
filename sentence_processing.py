"""
this module contatins the functions for pre-processing the sentences in the model, as described on our paper.

"""
import string
from nltk.stem import WordNetLemmatizer
import nltk.tag
import numpy as np

from definitions import *
from general_utils import increment_count, union_count_dicts

color_words = ["blue", "yellow", "black"]
shape_words = ["circle", "triangle", "square"]
integers = [str(i) for i in range(1,10)]
integer_words = "one two three four five six seven".split()
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
MIN_COUNT = 5



def load_vocabulary(filename):
    vocab = set()
    with open(filename) as vocab_file:
        for line in vocab_file:
            word = line.rstrip()

            vocab.add(word)
    return vocab


def load_ngrams(filename, n):
    ngrams = {}
    with open(filename) as ngrams_file:
        for line in ngrams_file:
            tokens = line.rstrip().split()
            counts = int(tokens[-1])
            ngram = tuple(tokens[:-1]) if n > 1 else tokens[0]
            ngrams[ngram] = counts
    return ngrams


def write_ngrams(filename, ngrams):
    with open(filename, 'w') as ngrams_file:
        for ngram, count in sorted(ngrams.items(), key= lambda entry : entry[1], reverse=True):
            ngram = [ngram] if type(ngram)==str else ngram
            count = str(count)
            line = " ".join([t for t in ngram] + [count])
            ngrams_file.write(line + '\n')
    return

def load_synonyms(file_name):
    result = {}
    with open(file_name) as syns:
        for line in syns:
            line = line.rstrip()
            if len(line)>1:
                sep = line.find(' ')
                token = line[:sep]
                result[token] = eval(line[sep+1 :])
    return result


def load_dict_from_txt(path):
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
            key = kvp[0].strip()
            value = kvp[1].strip()
            reps[key] = value
    return reps


def get_ngrams_counts(dataset, max_ngram, include_start_and_stop = False):
    """
        Gets a collection of sentences and extracts ngrams up to size n.
        each sentence is a list of words.
    """

    all_ngrams = []
    for i in range(max_ngram):
        all_ngrams.append(dict())

    for sentence in dataset:

        for word in sentence:
            increment_count(all_ngrams[0], word)
        if include_start_and_stop:
            sentence = ["<s>", "<s>"] + sentence;
            increment_count(all_ngrams[0], "<s>")
        length = len(sentence)
        for n in range(2, max_ngram+1):
            for i in range(n, length+1):
                increment_count(all_ngrams[n-1], tuple(sentence[i-n:i]))

    return all_ngrams




def get_sentence_ngram_logprob(sentence, p_dict):
    """
    return a probability vector over the next tokens given an ngram 'language model' of the
    the logical fforms. this is just a POC for generating plausible logical forms.
    the probability vector is the real model will come of course from the parameters of the
    decoder network.
    """
    sentence = sentence.split()
    unigram_counts, bigram_counts, trigram_counts, all_token_counts = p_dict
    lambda1= 0.4
    lambda2 = 0.4
    lambda3 = 1 - lambda1 - lambda2
    eval_token_count = 0
    sum_of_logs = 0
    tri_q = lambda wi_2, wi_1, wi: trigram_counts.get((wi_2, wi_1, wi), 0) / bigram_counts.get((wi_2, wi_1), 1)
    bi_q = lambda wi_1, wi: bigram_counts.get((wi_1, wi), 0) / unigram_counts[wi_1]
    uni_q = lambda wi: unigram_counts[wi] / all_token_counts

    sentence = ["<s>",'<s>'] + sentence
    for i in range(2, len(sentence)):
        wi_2, wi_1, wi = sentence[i - 2: i + 1]
        qLI = lambda1 * tri_q(wi_2, wi_1, wi) + \
              lambda2 * bi_q(wi_1, wi) + lambda3 * uni_q(wi)
        sum_of_logs += np.log2(qLI)

    return sum_of_logs / len(sentence)


def clean_sentence(sent):
    '''
    :param sent: a string
    :return: a copy of the original sentence, all in lower-case,
    without punctuation and without redundant whitespaces.
    '''
    s = str.lower(sent).strip()

    if not s.isalnum():
        s = s.translate(s.maketrans("","", string.punctuation))
        s = " ".join(s.split())
    return s


def variants(word, alphabet = ALPHABET):
    """get all possible variants for a word
    borrowed from https://github.com/pirate/spellchecker
    used for heuristic spelling correction of the dataset sentences
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts), splits


def rank_suggestion(suggested_token, prev_token, next_token, unigram_counts, bigram_counts, p = 0.1):
    # used for choosing to which word to replace a spelling errors, in case there is more than one suggestion,
    # using the one with highest ngram probability.

       return p * unigram_counts.get(suggested_token, 0) + \
              ((1 - p) / 2) * bigram_counts.get((prev_token, suggested_token),0) + \
              ((1 - p) / 2) * bigram_counts.get((suggested_token, next_token),0)


def replace_words_by_dictionary(sentences, dic):
    # sort by key length in decreasing order, to prevent override of previous changes
    manualy_chosen_replacements = sorted(dic.items(), key = lambda kvp : len(kvp[0].split()), reverse=True)
    manualy_chosen_replacements = [(" {} ".format(entry[0]) , " {} ".format(entry[1])) for entry in manualy_chosen_replacements]
    sentences_with_replacements = {}
    for k, s in sentences.items():
        s = " {} ".format(s) # pad with whitespaces
        for exp, replacement in manualy_chosen_replacements:
            s = s.replace(exp, replacement)
        sentences_with_replacements[k] = s.strip() # remove whitespaces

    return sentences_with_replacements


def replace_rare_words_with_unk(sentences, tokens_file=None):
    sentences = {k: s.split() for k, s in sentences.items()} # tokenize
    if not tokens_file:
        unigrams = get_ngrams_counts(sentences.values(), 1)[0]
    else:
        unigrams = load_ngrams(tokens_file, 1)
    for k, s in sentences.items():
        sentences[k] = " ".join([w  if unigrams.get(w,0)>=MIN_COUNT else '<UNK>' for w in s])
    return sentences


def preprocess_sentences(sentences_dic, mode = None, processing_type= None):
    """
    :param sentences_dic: dict mapping sentence ids to sentences
    :param mode: 'r' , 'w', or None
    :param processing_type: the depth of processing used. all processing
    phases are done sequentially one after the other.
    :return: r
    """
    # TODO right now it's either 'abstraction' *or* 'deep'. nee to amend here if we want both

    if processing_type is None:
        return sentences_dic # no processing

    if processing_type not in ("shallow", "spellproof", "lemmatize", "deep", "abstraction"):
        raise ValueError("Invalid processing type: {}".format(processing_type))

    if processing_type =='shallow': # just superficial processing
        return {k : clean_sentence(s) for k,s in sentences_dic.items()}

    # else continue to spelling correction

    tokenized_sentences = {k : str.split(clean_sentence(s)) for k,s in sentences_dic.items()}
    unigrams, bigrams = get_ngrams_counts(tokenized_sentences.values(), 2)
    lemmatizer = WordNetLemmatizer()

    vocab = load_vocabulary(ENG_VOCAB_60K)
    # delete one-character-words except for 'a'
    for ch in ALPHABET[1:]:
        vocab.discard(ch)
    for er in ('ad','al','tow','bo','bow','lease','lest', 'thee', 'bellow'):
        vocab.discard(er) # word from the English dictionary that are clearly not in our interest
    # add digits
    for dig in range(10):
        vocab.add(str(dig))

    unigrams_filtered = {token : count for token, count in unigrams.items() if token in vocab}
    bigrams_filtered =  {bigram : count for bigram, count in bigrams.items() if
                         bigram[0] in unigrams_filtered and bigram[1] in unigrams_filtered}

    if mode=='r':
        unigrams_filtered = union_count_dicts(unigrams_filtered, load_ngrams(TOKEN_COUNTS, 1))
        bigrams_filtered = union_count_dicts(bigrams_filtered, load_ngrams(BIGRAM_COUNTS, 2))


    corrections_inventory = {}

    for unigram in unigrams:
        if unigram not in unigrams_filtered:
            unigram_variants, bigram_variants = variants(unigram)
            unigram_corrections = [v for v in unigram_variants if v in unigrams_filtered]
            bigram_corrections = [bi for bi in bigram_variants if bi in bigrams_filtered]
            corrections_inventory[unigram] = \
                (unigram_corrections, bigram_corrections[0] if len(bigram_corrections)>0 else None)

    tagging = {}

    for k, tok_sent in tokenized_sentences.items():
        for i,w in enumerate(tok_sent):
            if w in corrections_inventory: # otherwise it is a valid word
                unigram_suggestions, bigram_suggestion =  corrections_inventory.get(w, None)
                if bigram_suggestion is not None: # if a bigram suggestion exists use it
                    tok_sent[i] = " ".join(bigram_suggestion)
                elif len(unigram_suggestions) == 1:
                    tok_sent[i] = unigram_suggestions[0]
                elif len(unigram_suggestions) > 1: # if several suggestions exits, use the one with the highest ranking
                    prev_token = (tok_sent[i-1].split(" "))[-1] if i>0 else "<s>"
                    next_token = tok_sent[i+1] if i<len(tok_sent)-1 else "<\s>"
                    tok_sent[i] = max(unigram_suggestions, key = lambda token : \
                            rank_suggestion(token, prev_token, next_token, unigrams_filtered, bigrams_filtered))
                else:
                    #print("could not resolve token "+ w)
                    if unigrams[w]<10: # only in training
                        tok_sent[i] = "<UNK>"
                if w in vocab:
                    pass#print ("{0}({1}): {2}".format(w, i, " ".join(tok_sent)))


    spellproofed_ss = {k : " ".join(s) for k,s in tokenized_sentences.items()}
    spellproofed_sentences = {k: s.split() for k, s in spellproofed_ss.items()}


    if mode == 'w':
        unigrams_checked, bigrams_checked = get_ngrams_counts(spellproofed_sentences.values(), 2)
        write_ngrams(TOKEN_COUNTS, unigrams_checked)
        write_ngrams(BIGRAM_COUNTS, bigrams_checked)

    if processing_type=='spellproof':
        return spellproofed_ss

    # else continue to lemmatization

    for k, s in spellproofed_sentences.items():

        tagging[k] = nltk.pos_tag(s)
        for i, w in enumerate(s):
            pos = tagging[k][i][1]
            if pos.startswith('N'):
                lemma = lemmatizer.lemmatize(w, 'n')
            elif pos.startswith('V'):
                lemma = lemmatizer.lemmatize(w, 'v')
            else:
                lemma = w
            s[i] = lemma if s[i] not in ('is', 'are') else s[i]


            if s[i] in integer_words and s[i]!= 'one':
                s[i] = integers[integer_words.index(s[i])]


    lemmatized_sentences =  {k: " ".join(s) for k, s in spellproofed_sentences.items()}
    if processing_type == 'lemmatize':
        return lemmatized_sentences

    # else move on to replacing words/phrases with others with similar meaning, in order to reduce vocabulary size

    if processing_type == 'abstraction':
        abstract_sentences = abstract(lemmatized_sentences)
        # lemmatized_sentences is of type {idx: sent}
        # abstract_sentences is of type {idx: (sent, rep_dict)}
        # rep_dict is of the form {'T_COLOR': 'yellow'}
        return abstract_sentences

    if SYNONYMS_PATH:
        replacements_dic = load_dict_from_txt(SYNONYMS_PATH)
        lemmatized_sentences_with_replacements = replace_words_by_dictionary(lemmatized_sentences, replacements_dic)


    # notice: to replace rare words with <unk> token, run the 'replace_rare_words_with_unk' method
    # on the output if this method

        return lemmatized_sentences_with_replacements

    return lemmatized_sentences

def abstract(sent_dict):
    '''
    :param sent_dict: {idx: sent}
    :return: {idx: (sent, {'T_COLOR': 'yellow"}}
    '''
    formalization_file = os.path.join(DATA_DIR, 'sentence-processing', 'formalized words.txt')
    words_to_patterns = load_dict_from_txt(formalization_file)
    for i in range(2, 9):
        words_to_patterns[str(i)] = 'T_INT'
    words_to_patterns["1"] = 'T_ONE'
    words_to_patterns["one"] = 'T_ONE'
    words_to_patterns['a single'] = 'T_ONE'

    for idx in sent_dict:
        rep_dict = {}
        sent = sent_dict[idx]
        for key in words_to_patterns.keys():
            if key in sent:
                if words_to_patterns[key] not in rep_dict.values():
                    sent = sent.replace(key, words_to_patterns[key])
                    rep_dict[words_to_patterns[key]] = key
                elif words_to_patterns[key] in rep_dict.values() and words_to_patterns[key]+'_1' not in rep_dict.values():
                    rep = words_to_patterns[key] + '_1'
                    sent = sent.replace(key, rep)
                    rep_dict[rep] = key
                elif words_to_patterns[key]+'_1' in rep_dict.values() and words_to_patterns[key]+'_2' not in rep_dict.values():
                    rep = words_to_patterns[key] + '_2'
                    sent = sent.replace(key, rep)
                    rep_dict[rep] = key
                else:
                    rep = words_to_patterns[key] + '_3'
                    sent = sent.replace(key, rep)
                    rep_dict[rep] = key

        sent_dict[idx] = (sent, rep_dict)

    return sent_dict
