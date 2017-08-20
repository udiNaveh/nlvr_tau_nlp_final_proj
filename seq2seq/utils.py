import random
from logical_forms_new import *

def epsilon_greedy_sample(choices, num_to_sample, epsilon=0.05):
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
    assert(len(choices) >= num_to_sample)
    assert(0 <= epsilon <= 1)

    if (len(choices) == num_to_sample):
        return choices

    # Performance
    if epsilon == 0:
        return choices[:num_to_sample]

    sample = []
    index_choices = range(len(choices))
    for i in range(num_to_sample):
        if random.random() <= epsilon or not i in index_choices:
            choice_index = random.choice(index_choices)
        else:
            choice_index = i
        index_choices.remove(choice_index)
        sample.append(choices[choice_index])
    return sample


def execute(program_tokens,image,token_mapping):
    reward = run_logical_form(process_token_sequence(program_tokens,token_mapping),image)
    return reward