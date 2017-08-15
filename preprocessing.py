import string

from handle_data import *
from general_utils import increment_count


ALPHABET = "abcdefghijklmnopqrstuvwxyz"



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
        for ngram, count in ngrams.items():
            ngram = [ngram] if type(ngram)==str else ngram
            count = str(count)
            line = " ".join([t for t in ngram] + [count])
            ngrams_file.write(line + '\n')
    return





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
            sentence = ["<s>"] + sentence; sentence.append("<\s>")
        length = len(sentence)
        for n in range(2, max_ngram+1):
            for i in range(n, length+1):
                increment_count(all_ngrams[n-1], tuple(sentence[i-n:i]))

    return all_ngrams


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
    """
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts), splits


def rank_suggestion(suggested_token, prev_token, next_token, unigram_counts, bigram_counts, p = 0.1):
       return p * unigram_counts[suggested_token] + \
              ((1 - p) / 2) * bigram_counts.get((prev_token, suggested_token),0) + \
              ((1 - p) / 2) * bigram_counts.get((suggested_token, next_token),0)


def preprocess_sentences(sentences_dic, mode = None):
    '''
    preprocesses the input sentences such that each returned sentences is a sequence of
    tokens that are part of the lexicon (???) seperated by whitespaces.
    using a hueristic approach for spell checking.

    :param sentences_dic: a dict[str, str] mapping from sentence identifier to sentence 
    :return: a dict[str, str] mapping from sentence identifier to sentence
    '''

    tokenized_sentences = {k : str.split(clean_sentence(s)) for k,s in sentences_dic.items()}
    unigrams, bigrams = get_ngrams_counts(tokenized_sentences.values(), 2)

    vocab = load_vocabulary(ENG_VOCAB_60K)
    # delete one-character-words except for 'a'
    for ch in ALPHABET[1:]:
        vocab.discard(ch)
    # add digits
    for dig in range(10):
        vocab.add(str(dig))

    if mode=='r':
        unigrams_filtered = load_ngrams(TOKEN_COUNTS, 1)
        bigrams_filtered = load_ngrams(BIGRAM_COUNTS, 2)
    else:
        unigrams_filtered = {token : count for token, count in unigrams.items()} # if token in vocab and count>=2}
        bigrams_filtered =  {kvp[0] : kvp[1] for kvp in bigrams.items() if
                             kvp[0][0] in unigrams_filtered and kvp[0][1] in unigrams_filtered}

    if mode == 'w':
        write_ngrams(TOKEN_COUNTS, unigrams_filtered)
        write_ngrams(BIGRAM_COUNTS, bigrams_filtered)

    # create an inventory of suggested corrections for invalid tokens from the sentences
    corrections_inventory = {}

    for unigram in unigrams:
        if unigram not in unigrams_filtered:
            unigram_variants, bigram_variants = variants(unigram)
            unigram_corrections = [v for v in unigram_variants if v in unigrams_filtered]
            bigram_corrections = [bi for bi in bigram_variants if bi in bigrams_filtered]
            corrections_inventory[unigram] = \
                (unigram_corrections, bigram_corrections[0] if len(bigram_corrections)>0 else None)

    for tok_sent in tokenized_sentences.values():
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
                    #print("could no resolve token "+ w)
                    if unigrams[w]<10: # only in training
                        tok_sent[i] = "<UNK>"

    return {k : " ".join(s) for k,s in tokenized_sentences.items()}

if __name__ == '__main__':
    data, sentences = build_data(read_data(TRAIN_JSON), preprocess=True)
    print()
    #preprocess_sentences(sentences)

