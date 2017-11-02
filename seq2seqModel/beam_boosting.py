"""
This module contains the functions for some of the techniques we used to improve the performance of the
beam search in learning and at test, by utilizing the structure or words of a sentence.
Specifically it contains the methods for keeping and using a cache of rewarded programs by their
patterns, and to re-rank the programs in the beam according to the tokens they include. 
"""


import definitions
from sentence_processing import *
from general_utils import *
from seq2seqModel.utils import *
from seq2seqModel.partial_program import *

#words_vocabulary = ["1","one","2","3","4","5","6","7","and","below","big","black","blue","bottom","circle","color","corner","different","edge","each","all","many","same","exactly","only","first","less","least","most","medium","middle","both","more","no","none","not","on","or","right","number","second","shape","size","small","square","stack","third","top","touch","triangle","yellow"]
words_vocabulary = ["there","is","a","circle","closely","touch","corner","of","box","are","2","yellow","block","<UNK>","blue","item","at","least","one","tower","with","exactly","3","1","5","black","and","only","square","it","the","wall","triangle","its","side","grey","object","top","not","edge","above","contain","as","4","first","from","base","which","have","over","no","bottom","attach","to","6","color","more","than","on","each","all","different","that","most","right","single","7","same","in","other","another","any","less","big","number","roof","both","where","medium","size","second","or","nearly","line","include","none","stack","together","third","contains","they","multiple","an","every","beneath","many","either","middle","small","height","set","shape","but","between","position","below","lot"]
tokens_vocabulary = ['OR', 'get_touching', 'le', 'All', 'ge', 'is_big', 'is_square', '5', 'Shape.SQUARE', 'Side.TOP', 'Color.BLUE', 'Color.YELLOW', 'get_below', '3', 'query_shape', '2', 'is_blue', '1', 'get_above', 'AND', 'Shape.TRIANGLE', 'Side.BOTTOM', 'query_color', 'count', 'is_third', 'is_black', 'Side.RIGHT', 'lt', 'and', 'all_same', 'is_touching_corner', 'NOT', 'is_small', '4', 'equal', 'select', 'Shape.CIRCLE', 'is_circle', 'is_top', '6', 'is_bottom', 'Color.BLACK', 'equal_int', '7', 'is_touching_wall', 'is_second', 'gt', 'is_yellow', 'is_triangle', 'query_size', 'is_medium']
words_array = np.array(words_vocabulary)
tokens_array = np.array(tokens_vocabulary)

log_dict = {'yellow': 'yellow', 'blue': 'blue', 'black': 'black', 'top': 'top', 'bottom': 'bottom',
            'exactly': 'equal_int', 'at least': 'le', 'at most': 'ge', 'triangle': 'triangle',
            'circle': 'circle', 'square': 'square', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
            '7': '7', '1': '1', 'one': '1', 'big' : 'big', 'small' : 'small', 'medium' : 'medium',
            'more than' : 'lt', 'less than' : 'gt', 'on': 'above', 'below': 'below', 'touch' : 'touching'}


formalization_file = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'formalized words.txt')
words_to_patterns = load_dict_from_txt(formalization_file)
for i in range(2, 10):
    words_to_patterns[str(i)] = 'T_INT'
words_to_patterns["1"] = 'T_ONE'
words_to_patterns["one"] = 'T_ONE'




def get_formalized_sentence(sentence):
    '''
    same as the one below but only for a sentence
    :param sentence: 'there is a yellow item'
    :return: 'there is a T_COLOR item'
    '''

    # building replacements "dictionary" (it is actually a list of tuples)

    manualy_chosen_replacements = sorted(words_to_patterns.items(), key = lambda x : sentence.find(x[0]))
    manualy_chosen_replacements = [(" {} ".format(entry[0]), " {} ".format(entry[1])) for entry in manualy_chosen_replacements]

    formalized_sentence = " {} ".format(sentence)  # pad with whitespaces
    used_reps = []
    # reminder: exp = yellow, replacement = T_COLOR
    for exp, replacement in manualy_chosen_replacements:
        if replacement not in used_reps and exp in formalized_sentence:
            formalized_sentence = formalized_sentence.replace(exp, replacement)
            used_reps.append(replacement)
        elif replacement in used_reps and exp in formalized_sentence and (replacement.rstrip() + '_1 ') not in used_reps:
            replacement = replacement.rstrip() + '_1 '
            formalized_sentence = formalized_sentence.replace(exp, replacement)
            used_reps.append(replacement)
        else:
            replacement = replacement.rstrip() + '_2 '
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

    matching_patterns = patterns_dict.get(formalized_sent, {})

    for i, word in enumerate(words):
        if i< len(words)-1:
            if word == 'at' and (words[i+1] == 'most' or words[i+1] == 'least'):
                words[i:i+2] = [' '.join(words[i:i+2])]
            if (word == 'more' or word == 'less') and  words[i+1] == 'than':
                words[i:i + 2] = [' '.join(words[i:i + 2])]


    suggested_decodings = []
    for prog, acc_reward in sorted(matching_patterns.items(), key = lambda item : binomial_prob(item[1][0],item[1][1])):
        token_seq = prog.split()

        for i, _ in enumerate(words):
            try:
                if words[i] == formalized_words[i]:
                    continue
            except IndexError:
                continue

            for j, token in enumerate(token_seq):
                if formalized_words[i] in token and _numbers_contained(formalized_words[i]) == _numbers_contained(token)\
                        and words[i] in log_dict:
                            formalized_token= token_seq[j]
                            rep = str.upper(log_dict[words[i]]) if '.' in formalized_token else log_dict[words[i]]
                            str.upper(token_seq[j]) if '.' in token_seq[j] else token_seq[j]
                            token_seq[j] =  (formalized_token).replace(formalized_words[i],  rep)
                            if token_seq[j] not in token_mapping:
                                print("token {} not exist".format( token_seq[j]))



        token_str = ' '.join(token_seq)
        suggested_decodings.append(token_str)
    return suggested_decodings



def _numbers_contained(string):
    nums = []
    for char in string:
        if char.isdigit():
            nums.append(char)
    return nums


def update_programs_cache(cached_programs, sentence, prog, prog_stats):
    '''
    :param sentence: 'there is a yellow item'
    :param program: exist filter ALL_ITEMS lambda_x_: is_yellow x
    :return:
            'there is a T_COLOR item', {'exist filter ALL_ITEMS lambda_x_: is_T_COLOR x': None}
            and adding both to patterns_dict
    '''
    token_seq = prog.token_seq if isinstance(prog, PartialProgram) else prog
    formalized_sentence = get_formalized_sentence(sentence)
    if formalized_sentence not in cached_programs:
        cached_programs[formalized_sentence] = {}
    matching_cached_patterns = cached_programs.get(formalized_sentence)


    manualy_chosen_replacements = sorted(words_to_patterns.items(), key = lambda x : sentence.find(x[0]))
    manualy_chosen_replacements = [(" {} ".format(entry[0]) , " {} ".format(entry[1])) for entry in manualy_chosen_replacements]
    formalized_sentence = " {} ".format(sentence)  # pad with whitespaces
    formalized_program = " {} ".format(" ".join(token_seq))  # pad with whitespaces

    temp_dict = {}
    for exp, replacement in manualy_chosen_replacements:
        if exp in formalized_sentence and replacement not in temp_dict.values():
            temp_dict[exp] = replacement
        elif exp in formalized_sentence and replacement in temp_dict.values() and (replacement.rstrip() + '_$ ') not in temp_dict.values():
            temp_dict[exp] = replacement.rstrip() + '_$ '
        elif exp in formalized_sentence and replacement in temp_dict.values() and (replacement.rstrip() + '_$ ') in temp_dict.values():
            temp_dict[exp] = replacement.rstrip() + '_2 '
    temp_dict = [(k, temp_dict[k]) for k in temp_dict]


    for exp, replacement in temp_dict:
        exp = exp.strip()
        if exp in log_dict: #
            formalized_program = formalized_program.replace(" {} ".format(log_dict[exp]), replacement)
            formalized_program = formalized_program.replace(str.upper(log_dict[exp]), replacement.strip())
            formalized_program = formalized_program.replace("_{}".format(log_dict[exp]), '_'+replacement.strip())
    formalized_program = formalized_program.strip()
    formalized_program = formalized_program.replace('$', '1')

    if formalized_program not in matching_cached_patterns:
        matching_cached_patterns[formalized_program] = [0,0]

    matching_cached_patterns[formalized_program][0] += prog_stats.n_correct
    matching_cached_patterns[formalized_program][1] += prog_stats.n_incorrect


    total_n_correct, total_n_incorrect = matching_cached_patterns[formalized_program]
    if total_n_incorrect>0 and (total_n_correct / total_n_incorrect) <3:
        del matching_cached_patterns [formalized_program]
    return


def sentence_program_relevance_score(sentence, program, words_to_tokens, recurring = False):
    relevant_tokens_found = 0
    relevant_tokens_needed = 0
    sentence_words = sentence.split()
    copies = {}
    for word in sentence_words:
        if sentence_words.count(word)>1:
            if word not in copies:
                copies[word] = program.token_seq.copy()
    for word in sentence_words:
        if word in words_to_tokens:
            relevant_tokens_needed+=1
            token_seq = copies.get(word, program.token_seq)
            for l in words_to_tokens[word]:
                if all(tok in token_seq for tok in l):
                    relevant_tokens_found+=1
                    if word in copies and recurring:
                        for tok in l:
                            token_seq.remove(tok)
                    break

    if relevant_tokens_needed == 0:
        return 0
    return relevant_tokens_found / relevant_tokens_needed


def get_features(sentence, program):
    logp = np.array([program.logprob])
    bow_words = np.sum([words_array == x for x in sentence.split()],axis = 0)
    bow_tokens = np.sum([tokens_array == x for x in program.token_seq],axis = 0)
    all_features = np.concatenate((logp,bow_words,bow_tokens))

    return all_features

def beam_reranker(sentence, programs, words_to_tokens):
    programs_c = [p for p in programs]
    return sorted(programs_c, key=lambda prog: ( - sentence_program_relevance_score(sentence, prog, words_to_tokens),
                                              -prog.logprob))

