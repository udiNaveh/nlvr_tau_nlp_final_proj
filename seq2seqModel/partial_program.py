
import numpy as np
import os
import re
import pickle
import random

from sentence_processing import get_ngrams_counts
import definitions
from sentence_processing import *
from general_utils import *
from seq2seqModel.utils import *
from data_manager import load_functions
from seq2seqModel.hyper_params import MAX_DECODING_LENGTH


token_mapping = load_functions(definitions.LOGICAL_TOKENS_MAPPING_PATH)

rand_dict = {'T_COLOR': ['yellow', 'blue', 'black'], 'T_COLOR_2': ['yellow', 'blue', 'black'], 'T_COLOR_3': ['yellow', 'blue', 'black'],
             'T_SHAPE': ['circle', 'square', 'triangle'], 'T_SHAPE_2': ['circle', 'square', 'triangle'], 'T_SHAPE_3': ['circle', 'square', 'triangle'],
             'T_LOC': ['top', 'bottom'], 'T_LOC_2': ['top', 'bottom'], 'T_QUANTITY_COMPARE': ['lt', 'gt', 'le', 'ge'],
             'T_QUANTITY_COMPARE_2': ['lt', 'gt', 'le', 'ge'], 'T_QUANTITY_COMPARE_3': ['lt', 'gt', 'le', 'ge'],
             'T_REL': ['above', 'below'], 'T_REL_2': ['above', 'below'], 'T_INT': [str(i) for i in range(9)],
             'T_INT_2': [str(i) for i in range(9)], 'T_INT_3': [str(i) for i in range(9)], 'T_SIZE': ['big', 'medium', 'small'],
             'T_SIZE_2': ['big', 'medium', 'small'], 'T_SIZE_3': ['big', 'medium', 'small']}


class PartialProgram:
    """
    a class the represents an instance of seq2seq decoder, regardless of the decoding model itself.
    Each instance of PartialProgram is used for outputing one sequence of tokens (aka logical form).
    In a beam search approach, any partial program that is constracted is represented by another decoder instance
    (I think, not sure about that... @TODO)
    """

    def __init__(self, lt_mapping=None):

        self.vars = [c for c in "xyzwuv"]
        self.stack = ["bool"]
        self.token_seq = []
        self.vars_in_use = {}
        self.logprobs = []
        self.logical_tokens_mapping = lt_mapping
        self.stack_history = [tuple(["bool"])]


    def __repr__(self):
        return "Partial program:(log p : {0:.2f}, sequence : {1})".format(self.logprob, " ".join(self.token_seq))

    def __len__(self):
        return len(self.token_seq)

    def __getitem__(self, key):
        return self.token_seq[key]

    def __iter__(self):
        for s in self.token_seq:
            yield  s

    def __contains__(self, item):
        return item in self.token_seq

    def copy(self):
        pp_copy = PartialProgram(self.logical_tokens_mapping)
        pp_copy.vars = self.vars.copy()
        pp_copy.stack = self.stack.copy()
        pp_copy.token_seq = self.token_seq.copy()
        pp_copy.vars_in_use = self.vars_in_use.copy()
        pp_copy.logprobs = self.logprobs.copy()
        pp_copy.stack_history = self.stack_history.copy()
        return pp_copy

    @property
    def logprob(self):
        return np.sum(self.logprobs)

    def get_possible_continuations(self):
        """
        :return: the set of valid tokens for the next of the program, according to the various constraints 
        (syntactical, sentence-driven or others) that are used.  
        """

        if len(self.stack) == 0:
            if self.token_seq[-1] == '<EOS>':
                return []
            return ["<EOS>"]  # end of decoding

        if len(self.token_seq)>=MAX_DECODING_LENGTH:
            return []

        next_type = self.stack[-1]
        # the next token must have a return type that matches next_type or to be itself an instance of next_type
        # for example, if next_type is 'int' than the next token can be an integer literal, like '2', or the name
        # of a functions that has return type int, like 'count'.


        if next_type.startswith("bool_func"):
            # in that case there is no choice - the only valid token is 'lambda'
            if len(self.vars) == 0:
                return [] # cannot recover
            var = self.vars[0]  # get a new letter to represent the var
            type_of_var = next_type[10:-1]
            if type_of_var == '?':
                raise TypeError("var in lambda function must be typed")
            # save the type of ver, and the current size of stack - the scope of the var depends on it.
            self.vars_in_use[var] = (TokenTypes(return_type=type_of_var, args_types=[], necessity=[]), len(self.stack)-1)
            next_token = 'lambda_{0}_:'.format(var)
            return [next_token]

        else:

            possible_continuations = [t for t, v in self.logical_tokens_mapping.items()
                                      if check_types(next_type, v.return_type)]
            possible_continuations.extend([var for var, (type_of_var, idx) in  self.vars_in_use.items()
                                           if check_types(next_type, type_of_var.return_type)])

            impossible_continuations = self.__get_impossible_continuations()
            return [c for c in possible_continuations if c not in impossible_continuations]



    def __get_impossible_continuations(self):
        impossible_continuations = []
        if self.token_seq and self.token_seq[-1] in self.logical_tokens_mapping:
            last = self.token_seq[-1]
            last_return_type, last_args_types, _ = self.logical_tokens_mapping[last]
            if len(last_args_types) == 1 and last_args_types[0].startswith('set'):
                impossible_continuations.extend([t for t, v in self.logical_tokens_mapping.items()
                                                               if not v.args_types])
                if last_return_type=='bool':
                    impossible_continuations.extend([t for t, v in self.vars_in_use.items()])

            if not last_args_types:
                impossible_continuations.extend([t for t, v in self.logical_tokens_mapping.items()
                                                              if not v.args_types and v.return_type == last_return_type])
            if len(self.stack)==3:
                impossible_continuations.extend([t for t, v in self.logical_tokens_mapping.items()
                                                if len(v.args_types)>1])

            if len(self.token_seq) >= 2 and all(tok == self.token_seq[-1] for tok in self.token_seq[-2:]):
                impossible_continuations.append(self.token_seq[-1])


            open_filter_scopes_begginnings = [start for start, end in self.filter_scopes() if not end]
            if last=='filter':
                impossible_continuations.extend(
                [self.token_seq[t+1] for t in open_filter_scopes_begginnings if t<len(self.token_seq)-1])

            if last == 'equal':
                impossible_continuations.extend([t for t, v in self.logical_tokens_mapping.items()
                                                if not ('Color' in v.return_type or 'Shape' in v.return_type)])

            # if 'select' in self.token_seq:
            #     impossible_continuations.append('select')
            #
            # if last == 'select':
            #     impossible_continuations.append([t for t, v in self.logical_tokens_mapping.items() if t not in ('2','3')])

        if ABSTRACTION:
            if 'T_QUANTITY_COMPARE' not in self.token_seq:
                impossible_continuations += ['T_QUANTITY_COMPARE_2']
            if 'T_QUANTITY_COMPARE_2' not in self.token_seq:
                impossible_continuations += ['T_QUANTITY_COMPARE_3']
            if 'get_T_REL' not in self.token_seq:
                impossible_continuations += ['get_T_REL_2']
            if ('is_T_COLOR' or 'Color.T_COLOR') not in self.token_seq:
                impossible_continuations += ['is_T_COLOR_2', 'Color.T_COLOR_2']
            if ('is_T_COLOR_2' or 'Color.T_COLOR_2') not in self.token_seq:
                impossible_continuations += ['is_T_COLOR_3', 'Color.T_COLOR_3']
            if ('is_T_SHAPE' or 'Shape.T_SHAPE') not in self.token_seq:
                impossible_continuations += ['is_T_SHAPE_2', 'Shape.T_SHAPE_2']
            if ('is_T_SHAPE_2' or 'Shape.T_SHAPE_2') not in self.token_seq:
                impossible_continuations += ['is_T_SHAPE_3', 'Shape.T_SHAPE_3']
            if ('is_T_LOC' or 'Side.T_LOC') not in self.token_seq:
                impossible_continuations += ['is_T_LOC_2', 'Side.T_LOC_2']
            if 'is_T_SIZE' not in self.token_seq:
                impossible_continuations += ['is_T_SIZE_2']
            if 'is_T_SIZE_2' not in self.token_seq:
                impossible_continuations += ['is_T_SIZE_3']


        return impossible_continuations


    def add_token(self, token, logprob):
        """
        :param token: the token to be added to the program. 
        :param logprob: the lof -probability of that token at that step, according to the model used.
        :return: True if the token was successfully added, False otherwise.
        """

        if len(self.stack)==0:
            if token == '<EOS>':
                self.token_seq.append(token)
                return True
            else:
                raise ValueError("cannot add token {} to program: stack is empty".format(token))

        next_type = self.stack.pop()
        if token.startswith('lambda'):
            self.token_seq.append(token)
            self.stack.append('bool')
            var = token[7]
            self.vars.remove(var)
        else:
            if token in self.logical_tokens_mapping:
                token_return_type,  token_args_types, token_necessity = self.logical_tokens_mapping[token]
            elif token in self.vars_in_use:
                token_return_type, token_args_types, _ = self.vars_in_use[token][0]
            else:
                raise ValueError("cannot add token {} to program: not a known function or var".format(token))
            t = disambiguate(next_type, token_return_type)
            s = disambiguate(token_return_type, next_type)
            if s is not None:
                # update jokers in stack:
                for i in range(len(self.stack) - 1, -1, -1):
                    if '?' in self.stack[i]:
                        self.stack[i] = self.stack[i].replace('?', s)
                    else:
                        break

            self.token_seq.append(token)
            args_to_stack = [arg if t is None else arg.replace('?', t) for arg in
                             token_args_types[::-1]]
            self.stack.extend(args_to_stack)



        for var, (type_of_var, idx) in [kvp for kvp in self.vars_in_use.items()]:
        # after popping the stack, check whether one of the added variables is no longer in scope.
            if len(self.stack) <= idx:
                if var in self.token_seq:
                    del self.vars_in_use[var]
                else:
                    # [rune sentence where a lambda expression on a var appears without that var in it.
                    return False
                break

        self.stack_history.append(tuple(self.stack))
        self.logprobs.append(logprob)

        #
        if len(self.token_seq) >= 3 and all(tok == self.token_seq[-1] for tok in self.token_seq[-3:]):
            return False

        bool_scopes_str = [" ".join(self.token_seq[start : end]) for start, end in self.boolean_scopes() if end]
        if len(set(bool_scopes_str)) < len(bool_scopes_str):
            return False

        return True

    def boolean_scopes(self):
        scopes = []
        for i in range(1, len(self.stack_history)):
            if self.stack_history[i] and self.stack_history[i][-1] == 'bool':
                end_of_exp = ([j for j in range (i +1 , len(self.stack_history)) if len(self.stack_history[j])
                               < len(self.stack_history[i])])
                if not end_of_exp:
                    scopes.append((i, None))
                else:
                    scopes.append((i, min(end_of_exp)))
        return scopes

    def filter_scopes(self):
        scopes = []
        for i in range(len(self.token_seq)):
            if self.token_seq[i]  == 'filter':
                end_of_exp = ([j for j in range (i +1 , len(self.stack_history)) if len(self.stack_history[j])
                               < len(self.stack_history[i])])
                if not end_of_exp:
                    scopes.append((i, None))
                else:
                    scopes.append((i, min(end_of_exp)))
        return scopes

    def get_prefix_program(self, index):

        prefix_program = PartialProgram(self.logical_tokens_mapping)
        for i, tok in enumerate(self.token_seq[:index]):
            valid_next_tokens = prefix_program.get_possible_continuations()
            if tok not in valid_next_tokens:
                raise ValueError(("{0} : {1} \n".format(self.token_seq, tok)))
            log_p = self.logprobs[i]
            prefix_program.add_token(tok, log_p)

        return prefix_program

    def create_unabstract_tokseq(self, abstraction_dict):
        '''
        :param abstraction_dict: {'yellow': 'T_COLOR'}
        :return:
        '''
        unabs_dict = {v: k for k,v in abstraction_dict.items()}
        new_token_seq = []
        for token in self.token_seq:
            if 'T' not in token or any([token == k for k in ['NOT', 'ALL_ITEMS', 'Side.RIGHT', 'Side.LEFT']]):
                new_token_seq.append(token)
            else:
                key = token[token.index('T'):]
                if key in unabs_dict:
                    rep = unabs_dict[key]
                else:
                    rep = random.choice(rand_dict[key])
                if '.' in token:
                    t = token.replace(key, rep.upper())
                else:
                    t = token.replace(key, rep)
                new_token_seq.append(t)
        self.unabstact_token_seq = new_token_seq



## utility methods for PartialProgram

def check_types(required, suggested):
    '''
    utility function to check whether the return type of a candidate token
    matches the type in the top of the stack.
    :param required: a str or a list of strings representing types in the logical forms.
    :param suggested: a string representing a type in the logical forms
    :return: True if the provided return type matches the required type that is in the top of the stack. 

    '''
    if '|' in required:  # if there are several possible required types (e.g. Item and set<Item>)
        return any([check_types(x, suggested) for x in required.split('|')])

    if required == suggested:
        return True
    if required == '?' or suggested == '?':  # wildcard
        return True
    if required.startswith('set') and suggested.startswith('set'):
        return check_types(required[4:-1], suggested[4:-1])
    if required.startswith('bool_func') and suggested.startswith('bool_func'):
        return check_types(required[10:-1], suggested[10:-1])

    return False


def disambiguate(typed, nontyped):
    """
    
    utility function that "reveals" the runtime type of a wildcard '?' for two types that where checked.
    e.g. for set<Item> and set<?> return 'Item'.
    :param typed: a str or a list of strings representing types in the logical forms.
    :param nontyped: a string representing a type in the logical forms
    :return: the type that replaces '?', or None if there is no such matching. 
    """
    if not isinstance(typed, str):
        raise ValueError('{} is not a string'.format(typed))
    if not isinstance(nontyped, str):
        raise ValueError('{} is not a string'.format(nontyped))
    if '|' in typed:
        types = typed.split('|')
        res = [disambiguate(t, nontyped) for t in types if disambiguate(t, nontyped)]
        if len(res) > 0:
            return res[0]
    if nontyped == "?":
        return typed
    if nontyped == "set<?>" and re.match(re.compile(r'set<.+>'), typed) is not None:
        return typed[4:-1]
    elif nontyped == "set<set<?>>" and re.match(re.compile(r'set<set.+>>'), typed) is not None:
        return typed[8:-2]
    else:
        return None


def program_from_token_sequence(next_token_probs_getter, token_seq, logical_tokens_mapping):
    """    
    :param next_token_probs_getter: a function for getting the next valid tokens and their probabilities 
    :param token_seq: 
    :param logical_tokens_mapping: 
    :return: partial_program: a PartialProgram object containing the ginev token sequence
            valid_tokens_history: which tokens were valid after every step
            greedy_choices: the next tokens with highest model probability at each step.
    """

    partial_program = PartialProgram(logical_tokens_mapping)
    valid_tokens_history = []
    greedy_choices = []

    for ind, tok in enumerate(token_seq):
        valid_next_tokens, probs_given_valid = \
            next_token_probs_getter(partial_program)
        valid_tokens_history.append(valid_next_tokens)
        if tok not in valid_next_tokens:
            raise ValueError(("{0} : {1}, {2} \n".format(token_seq, tok, ind)))
        p = probs_given_valid[valid_next_tokens.index(tok)]
        greedy_choices.append(valid_next_tokens[np.argmax(probs_given_valid)])
        partial_program.add_token(tok, np.log(p))
    return partial_program, (valid_tokens_history, greedy_choices)




