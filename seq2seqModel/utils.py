import random
import numpy as np
from collections import namedtuple
from logical_forms import *
from scipy.stats import binom




ProgramExecutionStats = namedtuple('ProgramExecutionStats', ['compiled', 'predicted_labels',
                                                               'is_consistent', 'n_correct', 'n_incorrect' ])


def get_program_execution_stats(token_seq, related_samples, logical_tokens_mapping, sentence = ''):
    actual_labels = np.array([sample.label for sample in related_samples])
    execution_results = np.array([execute(token_seq, sample.structured_rep, logical_tokens_mapping, sentence)
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


def binomial_prob(n_correct, n_incorrect):
    n_samples = n_correct+n_incorrect
    return sum(binom.pmf(x, n_samples, 0.5) for x in range(n_correct ,n_samples+1))