import random
import numpy as np
from collections import namedtuple
from logical_forms import *



def execute(program_tokens,image,token_mapping):
    logical_form = process_token_sequence(program_tokens, token_mapping)
    try:
        result = run_logical_form(logical_form,image)
    except (TypeError, SyntaxError, ValueError, AttributeError,
                RuntimeError, RecursionError, Exception, NotImplementedError) as err:
        result = None

    # if result is None:
    #     if input("go inside? ") =='y':
    #         result = run_logical_form(logical_form, image)
    return result

ProgramExecutionStats = namedtuple('ProgramExecutionStats', ['compiled', 'predicted_labels',
                                                               'is_consistent', 'n_correct', 'n_incorrect' ])


def get_program_execution_stats(token_seq, related_samples, logical_tokens_mapping):
    actual_labels = np.array([sample.label for sample in related_samples])
    execution_results = np.array([execute(token_seq, sample.structured_rep, logical_tokens_mapping)
                                  for sample in related_samples])
    prog_compiled = all(res is not None for res in execution_results)
    predicted_labels = [res if res is not None else True for res in execution_results]
    is_consistent = all(predicted_labels == actual_labels) and prog_compiled
    n_correct = sum(predicted_labels == actual_labels)
    n_incorrect = len(actual_labels) - n_correct

    return ProgramExecutionStats(prog_compiled, predicted_labels, is_consistent, n_correct, n_incorrect)

def programs_reranker(sentence, programs, words_to_tokens):
    """   
    :param sentence: a string
    :param programs: a list of unique PartialProgram instances
    :param words_to_tokens: a dictionary mapping words to lists of related tokens
    :return: a list containing the programs resorted according to their relevance to the sentence
    """

    sentence_words = sentence.split()
    needed_tokens = []
    for word in sentence_words:
        needed_tokens.append(words_to_tokens.get(word, []))
    progs_to_token_relevance_count = {}
    for prog in programs:
        n_releveant_tokens = 0
        for i in range(len(needed_tokens)):
            for t in needed_tokens[i]:
                if t in prog:
                    n_releveant_tokens += 1
                    break
        progs_to_token_relevance_count[prog] = n_releveant_tokens
    return sorted(programs, key=lambda prog: (-progs_to_token_relevance_count[prog], -prog.logprob))

def programs_reranker_2(sentence, programs, words_to_tokens):
    programs_c = [p for p in programs]
    return sorted(programs_c, key=lambda prog: (- sentence_program_relevance_score(sentence, prog, words_to_tokens),
                                              -prog.logprob))


def sentence_program_relevance_score(sentence, program, words_to_tokens):
    relevant_tokens_found = 0
    relevant_tokens_needed = 0
    sentence_words = sentence.split()
    for word in sentence_words:
        if word in words_to_tokens:
            relevant_tokens_needed+=1
            for l in words_to_tokens[word]:
                if all(tok in program for tok in l):
                    relevant_tokens_found+=1
                    break
    if relevant_tokens_needed == 0:
        return 0
    return relevant_tokens_found / relevant_tokens_needed



def one_hot(dim, index):
    v = np.zeros(dim)
    v[index] = 1.0
    return v

def sparse_vector_from_indices(dim, indices):
    assert (max(indices) < dim and min(indices)>=0 if indices else True)
    v = np.zeros(dim)
    for index in indices:
        v[index] = 1.0
    return v



def softmax(x, axis=0):
    """Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) == 2:
        # Matrix
        M,N = x.shape
        if axis == 0:
            v = np.exp(x - np.max(x,1).reshape((M,1)))
            x = v/np.sum(v,1).reshape((M,1))
        elif axis == 1:
            v = np.exp(x - np.max(x,0).reshape((1,N)))
            x = v/np.sum(v,0).reshape((1,N))
        ### END YOUR CODE
    else:
        # Vector
        v = np.exp(x - np.max(x))
        x = v/np.sum(v)

    assert x.shape == orig_shape
    return x




def get_probs_from_ngram_language_model(self, p_dict, possible_continuations):
    """
    return a probability vector over the next tokens given an ngram 'language model' of the
    the logical fforms. this is just a POC for generating plausible logical forms.
    the probability vector is the real model will come of course from the parameters of the
    decoder network.
    """
    probs = []
    prevprev_token = self.token_seq[-1] if len(self.token_seq) > 0 else "<s>"
    prev_token = self.token_seq[-2] if len(self.token_seq) > 1 else "<s>"
    token_counts = p_dict[0]
    bigram_counts = p_dict[1]
    trigram_counts = p_dict[2]
    for token in possible_continuations:
        probs.append(max(token_counts.get(token, 0) + 10 * bigram_counts.get((prev_token, token), 0) + \
                         9 * trigram_counts.get((prevprev_token, prev_token, token), 0), 1))
    return np.array(probs) / np.sum(probs)