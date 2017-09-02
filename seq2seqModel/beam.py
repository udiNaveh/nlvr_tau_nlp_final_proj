import numpy as np
from seq2seqModel.logical_forms_generation import *
from seq2seqModel.utils import epsilon_greedy_sample


max_decoding_length = 20
epsilon_greedy_p = 0.1
beam_size = 50

def e_greedy_randomized_beam_search(next_token_probs_getter, logical_tokens_mapping,
                                    original_sentence = None, epsilon = epsilon_greedy_p):

    if original_sentence:
        logical_tokens_mapping = sentence_relevant_logical_tokens(logical_tokens_mapping, original_sentence)

    beam = [PartialProgram(logical_tokens_mapping)]
    # create a beam of possible programs for sentence, the iteration continues while there are unfinished programs in beam and t < max_beam_steps

    for t in range(max_decoding_length):
        # if t>1 :
        #     sampled_prefixes = sample_decoding_prefixes(next_token_probs_getter, 5, t)
        #     beam.extend(sampled_prefixes)
        continuations = {}

        for partial_program in beam:
            if t > 0 and partial_program[-1] == '<EOS>':
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

        # choose the #beam_size programs and place them in the beam
        all_continuations_list = [c for p in continuations.values() for c in p]
        all_continuations_list.sort(key=lambda c: - c.logprob)
        beam = epsilon_greedy_sample(all_continuations_list, beam_size, continuations, epsilon)

        if all([prog.token_seq[-1] == '<EOS>' for prog in beam]):
            break  # if we have beam_size full programs, no need to keep searching

    beam = [prog for prog in beam if prog.token_seq[-1] == '<EOS>']  # otherwise won't compile and therefore no reward
    for prog in beam:
        prog.token_seq.pop(-1)  # take out the '<EOS>' token
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


def sentence_relevant_logical_tokens(logical_tokens_mapping, sentence):
    return {k: v for k, v in logical_tokens_mapping.items() if
                              (not v.necessity) or any([w in sentence.split() for w in v.necessity])}


def program_from_token_sequence(next_token_probs_getter, token_seq, logical_tokens_mapping):
    partial_program = PartialProgram(logical_tokens_mapping)
    valid_tokens_history = []
    greedy_choices = []

    for tok in token_seq:
        valid_next_tokens, probs_given_valid = \
            next_token_probs_getter(partial_program)
        valid_tokens_history.append(valid_next_tokens)
        if tok not in valid_next_tokens:
            raise ValueError(("{0} : {1} \n".format(token_seq, tok)))
        p = probs_given_valid[valid_next_tokens.index(tok)]
        greedy_choices.append(valid_next_tokens[np.argmax(probs_given_valid)])
        partial_program.add_token(tok, np.log(p))
    return partial_program, (valid_tokens_history, greedy_choices)