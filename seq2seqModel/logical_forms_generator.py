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

TOKEN_MAPPING = os.path.join(definitions.DATA_DIR, 'logical forms', 'token mapping')
PARSED_EXAMPLES_T = os.path.join(definitions.DATA_DIR, 'parsed sentences', 'parses for check as tokens')
LOGICAL_TOKENS_EMBEDDINGS_PATH = os.path.join(definitions.DATA_DIR, 'logical forms', 'logical_tokens_embeddings')
MAX_LENGTH = 35


TokenTypes = namedtuple('TokenTypes', ['return_type', 'args_types'])


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
            if len(entry) < 3 or not entry[1].isdigit() or int(entry[1]) != len(entry) - 3:
                print("could not parse function in line  {0}: {1}".format(i, line))
                # should use Warning instead
                continue
            token, return_type, args_types = entry[0], entry[-1], entry[2:-1]
            functions_dict[token] = TokenTypes(return_type=return_type, args_types=args_types)

        functions_dict.update({str(i): TokenTypes(return_type='int', args_types=[]) for i in range(1, 10)})

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
    prevprev_token = token_history[-1]
    prev_token = token_history[-2]
    token_counts = p_dict[0]
    bigram_counts = p_dict[1]
    trigram_counts = p_dict[2]
    for token in possible_continuations:
        probs.append(max(token_counts.get(token, 0) + 10 * bigram_counts.get((prev_token, token), 0) + \
                         9 * trigram_counts.get((prevprev_token, prev_token, token), 0), 1))
    return np.array(probs) / np.sum(probs)

class PartialProgram:
    """
    a class the represents an instance of seq2seq decoder, regardless of the decoding model itself.
    Each instance of PartialProgram is used for outputing one sequence of tokens (aka logical form).
    In a beam search approach, any partial program that is constracted is represented by another decoder instance
    (I think, not sure about that... @TODO)
    """

    def __init__(self, other=None):

        if other is None:
            self.vars = [c for c in "xyzwuv"]
            self.stack = ["bool"]
            self.token_seq = []
            self.vars_in_use = {}
            self.logprob = 0

        elif isinstance(other, PartialProgram):
            self.vars = other.vars.copy()
            self.stack = other.stack.copy()
            self.token_seq = other.token_seq.copy()
            self.vars_in_use = other.vars_in_use.copy()
            self.logprob = other.logprob

        else:
            raise ValueError

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

    def get_possible_continuations(self, function_dict):

        if len(self.stack) == 0:
            return ["<EOS>"]  # end of decoding

        next_type = self.stack[-1]
        # the next token must have a return type that matches next_type or to be itself an instance of next_type
        # for example, if next_type is 'int' than the next token can be an integer literal, like '2', or the name
        # of a functions that has return type int, like 'count'.

        for var, (type_of_var, idx) in [kvp for kvp in self.vars_in_use.items()]:
            # after popping the stack, check whether one of the added variables is no longer in scope.
            if len(self.stack) <= idx:
                del self.vars_in_use[var]

        if next_type.startswith("bool_func"):
            # in that case there is no choice - the only valid token is 'lambda'
            if len(self.vars) == 0:
                return [] # cannot recover
            var = self.vars.pop(0)  # get a new letter to represent the var
            type_of_var = next_type[10:-1]
            if type_of_var == '?':
                raise TypeError("var in lambda function must be typed")
            # save the type of ver, and the current size of stack - the scope of the var depends on it.
            self.vars_in_use[var] = (TokenTypes(return_type=type_of_var, args_types=[]), len(self.stack)-1)
            next_token = 'lambda_{0}_:'.format(var)
            return [next_token]

        else:

            possible_continuations = [t for t, v in function_dict.items()
                                      if check_types(next_type, v.return_type)]
            possible_continuations.extend([var for var, (type_of_var, idx) in  self.vars_in_use.items()
                                           if check_types(next_type, type_of_var.return_type)])
            return possible_continuations


    def add_token(self, token, logprob, function_dict):
        if len(self.stack)==0:
            if token == '<EOS>':
                self.token_seq.append(token)
                return
            else:
                raise ValueError("cannot add token {} to program: stack is empty".format(token))
        next_type = self.stack.pop()
        if token.startswith('lambda'):
            self.token_seq.append(token)
            self.stack.append('bool')
        else:
            if token in function_dict:
                token_return_type,  token_args_types = function_dict[token]
            elif token in self.vars_in_use:
                token_return_type, token_args_types = self.vars_in_use[token][0]
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

        self.logprob += logprob

    def choose_next_token(self, probabilities, function_dict):

        '''
        don't use this!  

        '''
        if len(self.stack) == 0:
            return "<EOS>"  # end of decoding

        if len(self.vars)==0 or len(self.token_seq)>MAX_LENGTH:
            raise RuntimeError("decoding passed the allowed length - doesn't seem to go anywhere good")


        next_type = self.stack.pop()
        # the next token must have a return type that matches next_type or to be itself an instance of next_type
        # for example, if next_type is 'int' than the next token can be an integer literal, like '2', or the name
        # of a functions that has return type int, like 'count'.

        for var, idx in [kvp for kvp in self.vars_in_use.items()]:
            # after popping the stack, check whether one of the added variables is no longer in scope.
            if len(self.stack) < idx:
                del self.vars_in_use[var]
                del function_dict[var]

        if next_type.startswith("bool_func"):
            # in that case there is no choice - the only valid token is 'lambda'
            var = self.vars.pop(0)  # get a new letter to represent the var
            self.vars_in_use[var] = len(self.stack)  # save the current size of stack -
            # the scope of the var dependes on it.
            next_token = 'lambda_{0}_:'.format(var)
            type_of_var = next_type[10:-1]
            if type_of_var == '?':
                raise TypeError("var in lambda function must be typed")
            function_dict[var] = TokenTypes(return_type=type_of_var, args_types=[])
            self.token_seq.append(next_token)
            self.stack.append('bool')
            return next_token

        else:
            # get the probability vector for the possible next tokens, given the current state of the the
            # decoder (and maybe other things)

            possible_continuations = [t for t, v in function_dict.items()
                                      if check_types(next_type, v.return_type)]
            if len(possible_continuations)==0:
                raise RuntimeError("decoder for stuck with no possible continuations")
            probs = self.get_probs_from_ngram_language_model(p_dict, possible_continuations)
            next_token = np.random.choice(possible_continuations, p=probs)
            t = disambiguate(next_type, function_dict[next_token].return_type)
            s = disambiguate(function_dict[next_token].return_type, next_type)
            if s is not None:
                # update jokers in stack:
                for i in range(len(self.stack) - 1, -1, -1):
                    if '?' in self.stack[i]:
                        self.stack[i] = self.stack[i].replace('?', s)
                    else:
                        break

            self.token_seq.append(next_token)
            args_to_stack = [arg if t is None else arg.replace('?', t) for arg in
                             function_dict[next_token].args_types[::-1]]
            self.stack.extend(args_to_stack)
            return next_token

        
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

    def generate_decoding(self):
        while True:
            next_token = self.choose_next_token(p_dict)
            if next_token == '<EOS>':
                break
        return self.token_seq

def generate_logical_forms(n):

    for _ in range(n):
        dec = PartialProgram()
        try:
            s = dec.generate_decoding()
            print(s) # prints as a token sequence, but you can convert it to normal form using process_token_sequence

        except RuntimeError as err:
            print (err)

def generate_mock_embeddings_file():
    # todo delete
    token_mapping = load_functions(TOKEN_MAPPING)
    d = {t : np.random.rand(12) for t in token_mapping.keys()}





if __name__ == "__main__":
    p_dict = get_probs_from_file(PARSED_EXAMPLES_T)
    token_mapping = load_functions(TOKEN_MAPPING)
    generate_logical_forms(10)

