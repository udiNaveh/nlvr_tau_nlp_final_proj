import random
import numpy as np

from logical_forms_new import *

def epsilon_greedy_sample(choices, num_to_sample, prefixes, epsilon=0.05):
    """Samples without replacement num_to_sample choices from choices
    where the ith choice is choices[i] with prob 1 - epsilon, and
    uniformly at random with prob epsilon
    Args:
        choices (list[Object]): a list of choices
        num_to_sample (int): number of things to sample
        epsilon (float): probability to deviate
    Returns:
        list[Object]: list of size num_to_sample choices
    """

    assert(0 <= epsilon <= 1)

    if (len(choices) <= num_to_sample):
        return choices

    # Performance
    if epsilon == 0:
        return choices[:num_to_sample]

    sample = []
    index_choices = [j for j in range(len(choices))]
    nonempty_prefixes = [conts for pref, conts in prefixes.items() if conts]
    choice_index = -1
    for i in range(num_to_sample):
        if random.random() <= epsilon or not i in index_choices:
            while True:
                prefix_conts = random.choice(nonempty_prefixes)
                prog = random.choice(prefix_conts)
                choice_index = choices.index(prog)
                if choice_index in index_choices:
                    break

        else:
            choice_index = i
        index_choices.remove(choice_index)
        sample.append(choices[choice_index])
    return sample


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