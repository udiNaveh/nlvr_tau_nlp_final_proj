"""
This module contains the basic code for running the e_greedy_randomized_beam_search that is used in our model.
"""


import numpy as np
import time
import random
from seq2seqModel.partial_program import *
from seq2seqModel.hyper_params import MAX_DECODING_LENGTH, MAX_STEPS, BEAM_SIZE, SKIP_AUTO_TOKENS, INJECT_TO_BEAM
from seq2seqModel.beam_boosting import *



decoding_steps_from_sentence_length = lambda n : 5 + n



def e_greedy_randomized_beam_search(next_token_probs_getter, logical_tokens_mapping,
                                    original_sentence = None, epsilon = 0.0, suggested_decodings = []):
    """
    An implementation of beam search for decoding logical forms.
    :param next_token_probs_getter: a function that returns the possible next token for a program at a given state,
            and their probabilities according to some model.
    :param logical_tokens_mapping:  a dictionary that maps logical tokens to their syntactic properties. needed for
            creating partial programs.
    :param (optional) original_sentence:  a string with the sentence whose logical form is desired. This can be used
             to restrict the search, but is not mandatory. It is assumed that the probabilistic model underlying 
             next_token_probs_getter is given the original sentence.             
    :param (optional) epsilon:  a real number in [0,1] representing the level of randomness in selecting the programs 
            for the beam from all possible continuations at each steps. if epsilon==0 (default) this is classic beam search.
            if epsilon==1 the selection is completely random.
    :param (optional) suggested_decodings: a list of suggested logical forms. These are based on programs that
            got reward on similar sentences.
            
    :return: The final beam. a list of <= beam_size partial programs with valid boolean logical forms, sorted
            by their model probability.
    """


    if original_sentence:
        # given the original sentence whose logical form is desired, limit the beam search according to it:
        # 1. Avoid certain tokens if they are not explicitly suggested from the sentence (e.g. do not allow 'is_yellow'
        #   if the word 'yellow' is not in the original sentence)
        # 2. limit the program length according to the sentence length

        logical_tokens_mapping = {k: v for k, v in logical_tokens_mapping.items() if
         (not v.necessity) or any([w in original_sentence.split() for w in v.necessity])}

        max_decoding_steps = decoding_steps_from_sentence_length(len(original_sentence.split()))
    else:
            max_decoding_steps = MAX_STEPS

    suggested_programs = {}
    for decoding in suggested_decodings:
        # if there are logical forms suggested for this sentence, create partial programs from them
        # to get their model probability. For each such programs save the indices in which
        # there were more than one valid option for the token. This information is needed later when
        # the programs are 'injected' to the beam at each decoding step.

        try:
            program, (valid_tokens_history, _) = program_from_token_sequence(
                next_token_probs_getter, decoding, logical_tokens_mapping)
            if SKIP_AUTO_TOKENS:
                checkpoints = [i for i in range(len(valid_tokens_history)) if len(valid_tokens_history[i])>1]
            else:
                checkpoints = [i for i in range(len(valid_tokens_history))]
            suggested_programs[program] = checkpoints
        except ValueError as err:
            # if for some reason a suggested decoding was not a valid program
            continue

    if not SKIP_AUTO_TOKENS:
        max_decoding_steps = MAX_DECODING_LENGTH
    beam = [PartialProgram(logical_tokens_mapping)] # initialize the beam with the empty program

    for t in range(max_decoding_steps):
        # at each step generate all possible continuations for programs in the beam and choose among them
        # the beam for the next step.

        beam_token_seqs = set([tuple(p.token_seq) for p in beam])

        if INJECT_TO_BEAM:
            # make sure that all prefixes of the suggested programs corresponding to t decoding steps in the
            # are in the beam.
            for prog, checkpoints in suggested_programs.items():
                if t>0 and t < len(checkpoints):
                    try:
                        ind = checkpoints[t-1] + 1
                        partial_seq = tuple(prog[: ind])
                        if not partial_seq in beam_token_seqs:
                            pp = prog.get_prefix_program(ind)
                            beam.append(pp)
                            beam_token_seqs.add(partial_seq)
                    except IndexError:
                        print("exception on prog {}".format(prog))

        continuations = {}

        for partial_program in beam:

            # if skip_autotokens is used, tokens that are the only valid continuation
            #  to a program ate automatically added to it. This means that at a given step, not all
            # programs in beam have the same number of tokens, but took the same number of choices to create them.
            bad_program= False
            while SKIP_AUTO_TOKENS:
                poss = partial_program.get_possible_continuations()
                if len(poss) == 1:
                    bad_program = not partial_program.add_token(poss[0], 0.0)
                    if bad_program:
                        break
                else:
                    break
            if bad_program:
                continue

            continuations[partial_program] = []

            # for complete programs add the program as is to the pool of possible continuations.
            # these programs may or may not survive until the end of the beam search.
            if t > 0 and  partial_program[-1] == '<EOS>':
                continuations[partial_program].append(partial_program)

            else:
                # otherwise, get the possible continuations for the program and add them to the pool
                # of all possible continuations in step t.
                valid_next_tokens, probs_given_valid = next_token_probs_getter(partial_program)
                logprob_given_valid = np.log(probs_given_valid)

                for i, next_tok in enumerate(valid_next_tokens):
                    pp = partial_program.copy()
                    if pp.add_token(next_tok, logprob_given_valid[i]):
                        continuations[partial_program].append(pp)

                if not continuations[partial_program]:
                    # programs that got stack with no possible continuations are
                    # not to be considered for selection to the next beam.
                    del continuations[partial_program]

        # stack together all possible continuations and sort them by model probability
        all_continuations_list = [c for p in continuations.values() for c in p]
        all_continuations_list.sort(key= lambda c: - c.logprob)
        # choose the beam_size continuations for the next step
        beam = epsilon_greedy_sample(all_continuations_list, BEAM_SIZE, epsilon)

        if all([prog.token_seq[-1] == '<EOS>' for prog in beam]):
            break  # if there are already beam_size complete programs in beam, no need to keep searching

    # filter out incomplete programs that won't compile anyway
    beam = [prog for prog in beam if prog.token_seq[-1] == '<EOS>']
    for prog in beam:
        prog.token_seq.pop(-1)  # take out the '<EOS>' token

    # make sure that all suggested programs are indeed in the final beam
    beam_token_seqs = [p.token_seq for p in beam]
    for prog in suggested_programs:
        if prog.token_seq not in beam_token_seqs:
            beam.append(prog)

    beam = sorted(beam, key=lambda prog: -prog.logprob)
    return beam


def epsilon_greedy_sample(choices, num_to_sample, epsilon):
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
    index_choices = [i for i in range(len(choices))]
    for i in range(num_to_sample):
        if random.random() <= epsilon or not i in index_choices:
            choice_index = random.choice(index_choices)
        else:
            choice_index = i
        index_choices.remove(choice_index)
        sample.append(choices[choice_index])
    return sample


def epsilon_greedy_sample_uniform_over_prefixes(choices, num_to_sample, prefixes, epsilon):
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

    assert (0 <= epsilon <= 1)

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
        if random.random() <= epsilon or i not in index_choices:
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





# sampling programs from the distribution (not using beam search) - not used in the model

def sample_valid_decodings(next_token_probs_getter, n_decodings, logical_tokens_mapping):
    decodings = []
    while len(decodings)<n_decodings:
        partial_program = PartialProgram(logical_tokens_mapping)
        for t in range(MAX_DECODING_LENGTH+1):
            if t > 0 and partial_program[-1] == '<EOS>':
                decodings.append(partial_program)
                break
            valid_next_tokens, probs_given_valid = \
                next_token_probs_getter(partial_program)
            if not valid_next_tokens:
                break
            next_token = np.random.choice(valid_next_tokens, p= probs_given_valid)
            p = probs_given_valid[valid_next_tokens.index(next_token)]
            partial_program.add_token(next_token, np.log(p))
    return decodings


def sample_decoding_prefixes(next_token_probs_getter, n_decodings, length, logical_tokens_mapping):
    decodings = []
    while len(decodings)<n_decodings:
        partial_program = PartialProgram(logical_tokens_mapping)
        for t in range(length):
            if t > 0 and partial_program[-1] == '<EOS>':
                decodings.append(partial_program)
                break
            valid_next_tokens, probs_given_valid = \
                next_token_probs_getter(partial_program)
            if not valid_next_tokens:
                break
            next_token = np.random.choice(valid_next_tokens, p= probs_given_valid)
            p = probs_given_valid[valid_next_tokens.index(next_token)]
            partial_program.add_token(next_token, np.log(p))
        decodings.append(partial_program)
    return decodings


