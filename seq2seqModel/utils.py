'''
utility methods used by seq2seq.py
'''

import random
import numpy as np
from collections import namedtuple
from logical_forms import *
from scipy.stats import binom
from general_utils import increment_count




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


def save_sentences_test_results(results_by_sentence, dataset, path):
    with open(path, 'w') as f:
        for sent_id, stats in results_by_sentence.items():

            sentence = dataset.get_sentence_by_id(sent_id, original=True)
            print("sentence: " + sentence, file=f)
            predicted_program = stats['top_program_by_reranking']
            print("model prediction:", file=f)
            print("log_prob = {0:.2f}, program = {1}".
                   format(predicted_program.logprob, " ".join(predicted_program.token_seq)), file=f)
            if stats['top_by_reranking_stats'].is_consistent:
                print("consistent", file=f)
            else:
                print("inconsistent", file=f)
                if not stats['consistent_programs']:
                    print ("no consistent programs in beam", file=f)
                else:
                    consistent_prog = max(stats['consistent_programs'], key = lambda p : p.logprob)
                    print("consistent program from beam : ", file=f)
                    print("log_prob = {0:.2f}, program = {1}".
                        format(consistent_prog.logprob, " ".join(consistent_prog.token_seq)), file=f)
            print("##############\n", file=f)



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

    else:
        # Vector
        v = np.exp(x - np.max(x))
        x = v/np.sum(v)

    assert x.shape == orig_shape
    return x


def binomial_prob(n_correct, n_incorrect):
    """
     return the probability for n_correct or more answers out of n+correct+n_incorrect.
    """
    n_samples = n_correct+n_incorrect
    return sum(binom.pmf(x, n_samples, 0.5) for x in range(n_correct ,n_samples+1))