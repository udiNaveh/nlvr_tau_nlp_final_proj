import definitions
from sentence_processing import *
from general_utils import *
from seq2seqModel.utils import *
from seq2seqModel.logical_forms_generation import *




log_dict = {'yellow': 'yellow', 'blue': 'blue', 'black': 'black', 'top': 'top', 'bottom': 'bottom',
            'exactly': 'equal_int', 'at least': 'le', 'at most': 'ge', 'triangle': 'triangle',
            'circle': 'circle', 'square': 'square', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
            '7': '7', '1': '1', 'one': '1', 'big' : 'big', 'small' : 'small', 'medium' : 'medium',
            'more than' : 'lt', 'less than' : 'gt', 'on': 'above', 'below': 'below', 'touch' : 'touching'}

# building replacements "dictionary" (it is actually a list of tuples)
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
                if formalized_words[i] in token and numbers_contained(formalized_words[i]) == numbers_contained(token)\
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



def numbers_contained(string):

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