import numpy as np
import time
from seq2seqModel.logical_forms_generation import *
from seq2seqModel.utils import epsilon_greedy_sample


max_decoding_length = 20
max_decoding_steps = 14
epsilon_greedy_p = 0.1
beam_size = 40
skip_autotokens = True
decoding_steps_from_sentence_length = lambda n : 5 + n
injection = True


def e_greedy_randomized_beam_search(next_token_probs_getter, logical_tokens_mapping,
                                    original_sentence = None, epsilon = epsilon_greedy_p, suggested_decodings = []):

    if original_sentence:
        logical_tokens_mapping = sentence_relevant_logical_tokens(logical_tokens_mapping, original_sentence)
        if decoding_steps_from_sentence_length:
            max_decoding_steps = decoding_steps_from_sentence_length(len(original_sentence.split()))

    else:
        max_decoding_steps = 14


    suggested_programs = {}

    time_added = 0.0
    if injection:
        start = time.time()
        for decoding in suggested_decodings:
            try:
                program, (valid_tokens_history, greedy_choices) = program_from_token_sequence(
                    next_token_probs_getter, decoding, logical_tokens_mapping, original_sentence=original_sentence)
                checkpoints = [i for i in range(len(valid_tokens_history)) if len(valid_tokens_history[i])>1]
                suggested_programs[program] = checkpoints
            except ValueError:
                continue
        end = time.time()
        time_added += end-start


    beam = [PartialProgram(logical_tokens_mapping)]
    # create a beam of possible programs for sentence, the iteration continues while there are unfinished programs
    #  in beam and t < max_beam_steps


    for t in range(max_decoding_steps):
        beam_token_seqs = [p.token_seq for p in beam]
        continuations = {}

        start = time.time()
        if t>0:
            for prog, checkpoints in suggested_programs.items():
                if t-1 < len(checkpoints):
                    try:
                        ind = checkpoints[t-1] + 1
                        partial_seq = prog[: ind]
                        if not partial_seq in beam_token_seqs:
                            beam_token_seqs.append(partial_seq)
                            pp = prog.get_prefix_program(ind)
                            beam.append(pp)
                    except IndexError:
                        print("exception on prog {}".format(prog))

        end = time.time()
        time_added+= (end-start)

        for partial_program in beam:
            bad_program= False
            while skip_autotokens:
                poss = partial_program.get_possible_continuations()
                if len(poss) == 1:
                    bad_program = not partial_program.add_token(poss[0], 0.0)
                    if bad_program:
                        break
                else:
                    break
            if bad_program:
                continue
            if t > 0:
                if partial_program[-1] == '<EOS>':
                    continuations[partial_program] = [partial_program]
                    continue
            cont_list = []

            valid_next_tokens, probs_given_valid = \
                next_token_probs_getter(partial_program)

            logprob_given_valid = np.log(probs_given_valid)

            for i, next_tok in enumerate(valid_next_tokens):
                pp = partial_program.copy()
                if pp.add_token(next_tok, logprob_given_valid[i]):
                    cont_list.append(pp)
            continuations[partial_program] = cont_list

        # choose the beam_size programs and place them in the beam
        all_continuations_list = [c for p in continuations.values() for c in p]
        all_continuations_list.sort(key=lambda c: - c.logprob)
        beam = epsilon_greedy_sample(all_continuations_list, beam_size, continuations, epsilon)

        if all([prog.token_seq[-1] == '<EOS>' for prog in beam]):
            break  # if we have beam_size full programs, no need to keep searching

    beam = [prog for prog in beam if prog.token_seq[-1] == '<EOS>']  # otherwise won't compile and therefore no reward
    for prog in beam:
        prog.token_seq.pop(-1)  # take out the '<EOS>' token

    if injection:
        beam_token_seqs = [p.token_seq for p in beam]
        for prog in suggested_programs:
            if prog.token_seq not in beam_token_seqs:
                beam.append(prog)

    beam = sorted(beam, key=lambda prog: -prog.logprob)
    return beam




def e_greedy_randomized_beam_search_omer(next_token_probs_getter, logical_tokens_mapping, suggested_progs,
                                         original_sentence=None, epsilon=epsilon_greedy_p):
    '''
    :param next_token_probs_getter:
    :param logical_tokens_mapping:
    :param suggested_progs: list of tokens lists
    :param original_sentence:
    :param epsilon:
    :return:
    '''

    if original_sentence:
        logical_tokens_mapping = sentence_relevant_logical_tokens(logical_tokens_mapping, original_sentence)
        if decoding_steps_from_sentence_length:
            max_decoding_steps = decoding_steps_from_sentence_length(len(original_sentence.split()))
        else:
            max_decoding_steps = 14

    beam = [PartialProgram(logical_tokens_mapping)]
    # create a beam of possible programs for sentence, the iteration continues while there are unfinished programs in beam and t < max_beam_steps

    if injection:
        suggested_partial_progs = [PartialProgram(logical_tokens_mapping)]*len(suggested_progs)

    for t in range(max_decoding_steps):
        # t steps in the beam search

        # getting all possible continuations and their probs
        # continuations - dict of lists {pp: [conts]}
        continuations = {}
        for partial_program in beam:
            BP = False
            while skip_autotokens:
                poss = partial_program.get_possible_continuations()
                if len(poss) == 1:
                    BP = not partial_program.add_token(poss[0], 0.0)  # what's that?
                else:
                    break
            if BP:
                continuations[partial_program] = []
                continue
            if t > 0:
                if partial_program[-1] == '<EOS>':
                    continuations[partial_program] = [partial_program]
                    continue
            cont_list = []

            valid_next_tokens, probs_given_valid = \
                next_token_probs_getter(partial_program)

            logprob_given_valid = np.log(probs_given_valid)

            for i, next_tok in enumerate(valid_next_tokens):
                pp = partial_program.copy()
                pp.add_token(next_tok, logprob_given_valid[i])
                cont_list.append(pp)
            continuations[partial_program] = cont_list

        # choose the beam_size programs and place them in the beam
        all_continuations_list = [c for p in continuations.values() for c in p]

        if injection:
            for i, prog in enumerate(suggested_partial_progs):
                if len(suggested_progs[i]) <= t:
                    continue
                tokens, probs = next_token_probs_getter(suggested_partial_progs[i])
                prog.add_token(suggested_progs[i][t], np.log(probs[tokens.index(suggested_progs[i][t])]))
                if prog.token_seq not in [p.token_seq for p in all_continuations_list]:
                    all_continuations_list.append(prog)

        all_continuations_list.sort(key=lambda c: - c.logprob)
        beam = epsilon_greedy_sample(all_continuations_list, beam_size, continuations, epsilon)

        # assert(len(prog.token_seq)<=t+1 for prog in beam)

        if all([prog.token_seq[-1] == '<EOS>' for prog in beam]):
            break  # if we have beam_size full programs, no need to keep searching

    beam = [prog for prog in beam if prog.token_seq[-1] == '<EOS>']  # otherwise won't compile and therefore no reward
    for prog in beam:
        prog.token_seq.pop(-1)  # take out the '<EOS>' token

    # adding the suggested programs to the beam in the end.
    # TODO important! if it is done outside of this function, erase the following if-clause
    if injection:
        for prog in suggested_progs:
            sp, _ = program_from_token_sequence(next_token_probs_getter, prog, logical_tokens_mapping,
                                                original_sentence=original_sentence)
            if sp.token_seq not in [p.token_seq for p in beam]:
                beam.append(sp)

    beam = sorted(beam, key=lambda prog: -prog.logprob)
    return beam


def sample_valid_decodings(next_token_probs_getter, n_decodings, logical_tokens_mapping):
    decodings = []
    while len(decodings)<n_decodings:
        partial_program = PartialProgram(logical_tokens_mapping)
        for t in range(max_decoding_length+1):
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


