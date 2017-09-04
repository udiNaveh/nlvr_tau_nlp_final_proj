'''
decoder poc
'''

from collections import namedtuple
import numpy as np
import os
import re


from preprocessing import get_ngrams_counts
import definitions
from logical_forms_new import process_token_sequence
from preprocessing import *
from general_utils import *

TOKEN_MAPPING = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping_limitations')
PARSED_EXAMPLES_T = os.path.join(definitions.DATA_DIR, 'parsed sentences', 'parses for check as tokens')
LOGICAL_TOKENS_EMBEDDINGS_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'logical_tokens_embeddings')
MAX_LENGTH = 35

USE_PARAPHRASING = False

TokenTypes = namedtuple('TokenTypes', ['return_type', 'args_types', 'necessity'])


def get_probs_from_file(path):
    token_seqs = []
    with open(path) as parsed_examples:
        for line in parsed_examples:
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith('#'):
                continue
            if not line[0].isdigit():
                token_seqs.append(line.split())

    return get_ngrams_counts(token_seqs, 3, include_start_and_stop=True)


def load_functions(filename):
    """
    loads from file a dictionary of all valid tokens in the formal language we use.
    each token is is defines by its name, its return types, and its argument types.
    tokens that represent known entities, like ALL_BOXES, or Color.BLUE are treated as
    functions that take no arguments, and their return type is their own type, i.e. 
    set<set<Item>>, and Color, correspondingly.
    """
    functions_dict = {}
    with open(filename) as functions_file:
        for i, line in enumerate(functions_file):
            if line.isspace():
                continue
            line = line.strip()
            if line.startswith('#'):
                continue
            entry = line.split()

            split_idx = entry.index(':') if ':' in entry else len(entry)
            entry, necessary_words = entry[:split_idx], entry[split_idx:]

            if len(entry) < 3 or not entry[1].isdigit() or int(entry[1]) != len(entry) - 3:
                print("could not parse function in line  {0}: {1}".format(i, line))
                # should use Warning instead
                continue
            token, return_type, args_types = entry[0], entry[-1], entry[2:-1]
            functions_dict[token] = TokenTypes(return_type=return_type, args_types=args_types, necessity= necessary_words)
        functions_dict['1'] = TokenTypes(return_type='int', args_types=[], necessity=['1', 'one', 'a', 'is'])
        functions_dict.update({str(i): TokenTypes(return_type='int', args_types=[], necessity=[str(i)]) for i in range(2, 10)})

    return functions_dict


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
    '''
    utility function that "reveals" the runtime type of a wildcard '?' for to typed that where checked.
    e.g. for set<Item> and set<?> return 'Item'.
    :param typed: a str or a list of strings representing types in the logical forms.
    :param nontyped: a string representing a type in the logical forms
    :return: the type that replaces '?', or None if there is no such matching. 
    '''
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


def get_ngram_probs(token_history, p_dict, possible_continuations):
    """
    return a probability vector over the next tokens given an ngram 'language model' of the
    the logical fforms. this is just a POC for generating plausible logical forms.
    the probability vector is the real model will come of course from the parameters of the
    decoder network.
    """
    probs = []
    prevprev_token = token_history[-2]
    prev_token = token_history[-1]
    token_counts = p_dict[0]
    bigram_counts = p_dict[1]
    trigram_counts = p_dict[2]
    for token in possible_continuations:
        probs.append(max(token_counts.get(token, 0) + 5 * bigram_counts.get((prev_token, token), 0) + \
                         9 * trigram_counts.get((prevprev_token, prev_token, token), 0), 1))
    return np.array(probs) / np.sum(probs)

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
        self.logprob = 0
        self.logical_tokens_mapping = lt_mapping
        self.stack_history = []


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
        pp_copy.vars = self.vars
        pp_copy.vars = self.vars.copy()
        pp_copy.stack = self.stack.copy()
        pp_copy.token_seq = self.token_seq.copy()
        pp_copy.vars_in_use = self.vars_in_use.copy()
        pp_copy.logprob = self.logprob
        pp_copy.stack_history = self.stack_history.copy()
        return pp_copy

    def get_possible_continuations(self):

        if len(self.stack) == 0:
            if self.token_seq[-1] == '<EOS>':
                return []
            return ["<EOS>"]  # end of decoding


        if len(self.token_seq) >= 3 and all(tok == self.token_seq[-1] for tok in self.token_seq[-3:]):
            return []

        if len(self.stack) >4:
            return []


        next_type = self.stack[-1]
        # the next token must have a return type that matches next_type or to be itself an instance of next_type
        # for example, if next_type is 'int' than the next token can be an integer literal, like '2', or the name
        # of a functions that has return type int, like 'count'.



        if next_type.startswith("bool_func"):
            # in that case there is no choice - the only valid token is 'lambda'
            if len(self.vars) == 0:
                return [] # cannot recover
            var = self.vars.pop(0)  # get a new letter to represent the var
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
            # if str.isdigit(last):
            # impossible_continuations.extend([str(i) for i in range(10)])
            last_return_type, last_args_types, _ = self.logical_tokens_mapping[last]
            if len(last_args_types) == 1 and last_args_types[0].startswith('set'):
                impossible_continuations.extend([t for t, v in self.logical_tokens_mapping.items()
                                                               if not v.args_types])
            if not last_args_types:
                impossible_continuations.extend([t for t, v in self.logical_tokens_mapping.items()
                                                              if not v.args_types and v.return_type == last_return_type])
            if len(self.stack)==3:
                impossible_continuations.extend([t for t, v in self.logical_tokens_mapping.items()
                                                if len(v.args_types)>1])

        return impossible_continuations


    def add_token(self, token, logprob):
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
                    return False
                break


        self.logprob += logprob
        self.stack_history.append(tuple(self.stack))
        return True


def get_formalized_sentence(sentence):
    '''
    same as the one below but only for a sentence
    :param sentence: 'there is a yellow item'
    :return: 'there is a T_COLOR item'
    '''
    formalization_file = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'formalized words.txt')
    dict = load_dict_from_txt(formalization_file)
    for i in range(2,10):
        dict[str(i)] = 'T_INT'
    dict["1"] = 'T_ONE'
    dict["one"] = 'T_ONE'
    manualy_chosen_replacements = sorted(dict.items(), key = lambda kvp : len(kvp[0].split()), reverse=True)
    # manualy_chosen_replacements = [("{}".format(entry[0]), "{}".format(entry[1])) for entry in manualy_chosen_replacements]
    manualy_chosen_replacements = [(" {} ".format(entry[0]) , " {} ".format(entry[1])) for entry in manualy_chosen_replacements]
    formalized_sentence = " {} ".format(sentence)  # pad with whitespaces
    # used_replacements = {}
    # words = sentence.split()
    # for word in words:
    #     if word in manualy_chosen_replacements:
    #         increment_count(used_replacements, manualy_chosen_replacements[word])
    # for i, word in enumerate(words):
    #     if word not in used_replacements:
    #         continue
    #     elif used_replacements[manualy_chosen_replacements[word]] == 1:
    #         words[i] = manualy_chosen_replacements[word]
    #     else:

    for exp, replacement in manualy_chosen_replacements:
        formalized_sentence = formalized_sentence.replace(exp, replacement)
    formalized_sentence = formalized_sentence.strip()

    return formalized_sentence

def get_programs_for_sentence_by_pattern(sentence, patterns_dict):
    '''
    :param sentence: english sentence, str
    :param patterns_dict: dict of english formalized sents and formalized logical forms, {str: str}
    :return: a *string* that is a suggested program based on the dict
    '''
    words = sentence.split()
    formalized_sent = get_formalized_sentence(sentence)
    formalized_words = formalized_sent.split()

    if formalized_sent not in patterns_dict:
        return None
    token_str = patterns_dict[formalized_sent]
    token_seq = token_str.split()

    for i in range(len(words)):
        if words[i] == formalized_words[i]:
            continue
        for j, token in enumerate(token_seq):
            if formalized_words[i] in token:
                token_seq[j] = words[i]

    token_str = ' '.join(token_seq)
    return token_str
    # return suggested_programs

def get_formlized_sentence_and_docding(sentence, program, patterns_dict):
    '''
    #todo figure out what to do when 'formal' tokens do not map to 'formal' words or vice versa 
    :param sentence: 
    :param program: 
    :return: 
    '''
    #TODO: OMER!
    formalized_sentence, formalized_decoding = [], []

    if get_formalized_sentence(sentence) in patterns_dict:
        return patterns_dict[get_formalized_sentence(sentence)]

    # building replacements "dictionary" (it is actually a list of tuples)
    formalization_file = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'formalized words.txt')
    dict = load_dict_from_txt(formalization_file)
    for i in range(2,10):
        dict[str(i)] = 'T_INT'
    dict["1"] = 'T_ONE'
    dict["one"] = 'T_ONE'
    manualy_chosen_replacements = sorted(dict.items(), key = lambda kvp : len(kvp[0].split()), reverse=True)
    manualy_chosen_replacements = [(" {} ".format(entry[0]) , " {} ".format(entry[1])) for entry in manualy_chosen_replacements]
    formalized_sentence = " {} ".format(sentence)  # pad with whitespaces
    formalized_program = " {} ".format(program)  # pad with whitespaces

    words = sentence.split()
    tokens = program.split()

    # # building a temporary dictionary, per sentence
    # temp_dict = {}
    # for word in words:
    #     if word in dict and word not in temp_dict:
    #         temp_dict[word] = dict[word]
    #     elif word in dict and word in temp_dict:
    #         temp_dict[word] = dict[word] + '_1'
    # for i, _ in enumerate(words):
    #     if words[i] in temp_dict:

    # reminder: exp = yellow, replacement = T_COLOR
    temp_dict = {}
    for exp, replacement in manualy_chosen_replacements:
        if exp in formalized_sentence and exp not in temp_dict:
            temp_dict[exp] = replacement
        elif exp in formalized_sentence and replacement in temp_dict.values():
            temp_dict[exp] = replacement.rstrip() + '_1 '
        elif exp in formalized_sentence and replacement in temp_dict.values() and (replacement.rstrip() + '_1 ') in temp_dict.values():
            temp_dict[exp] = replacement.rstrip() + '_2 '
    temp_dict = [(k, temp_dict[k]) for k in temp_dict]

    for exp, replacement in temp_dict:
        formalized_sentence = formalized_sentence.replace(exp, replacement)
    formalized_sentence = formalized_sentence.strip()

    log_dict = {' yellow ': 'yellow', ' blue ': 'blue', ' black ': 'black', ' top ': ' top ', ' bottom ': 'bottom',
                ' exactly ': 'equal_int', ' at least ': 'ge', ' at most ': 'le', ' triangle ': 'triangle',
                ' circle ': 'circle', ' square ': 'square', ' 2 ': '2', ' 3 ': '3', ' 4 ': '4', ' 5 ': '5', ' 6 ': '6',
                ' 7 ': '7', ' 1 ': '1', ' one ': '1'}

    for exp, replacement in temp_dict:
        formalized_program = formalized_program.replace(log_dict[exp], replacement.strip())
    formalized_program = formalized_program.strip()

    patterns_dict[formalized_sentence] = formalized_program

    return formalized_sentence, formalized_program

def numbers_contained(string):

    nums = []
    for char in string:
        if char.isdigit():
            nums.append(char)
    return nums

def update_programs_cache(cached_programs, sentence, prog, reward):
    pass
