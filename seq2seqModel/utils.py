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

# History embedding
def get_history_embedding(history, history_length, STACK=0):
    if STACK:
        # TODO
        raise NotImplementedError
    else:
        # TOKEN implementation : concat #history_length last tokens.
        # if the current history is shorter than #history_length, pad with zero vectors
        if len(history) < history_length:
            result = []
            diff = history_length - len(history)
            for i in range(history_length):
                if i < diff:
                    result = np.concatenate([result, np.zeros([12])], 0)
            if len(history) != 0:
                result = np.concatenate([result, history], 0)
            return result
        result = []
        for i in range(1, history_length+1):
            result = np.concatenate([result,history[-i]], 0)
        return result