import pickle
from seq2seqModel.logical_forms_generation import get_programs_for_sentence_by_pattern, get_formalized_sentence, get_formlized_sentence_and_decoding

i = open(r'pre-training\temp_sents_backup.txt', 'r')
o = open('patterns_dict', 'wb')

dict = {}
for line in i:
    if line.startswith('@'):
        key = line.split('$')[0][2:].rstrip()
        dict[key] = {}
        continue
    elif line.startswith('~'):
        value = line[2:].rstrip()
        dict[key][value] = None
    else:
        continue
    # if key not in dict:
    #     dict[key] = value

pickle.dump(dict,o)
o.close()

dict = pickle.load(open('patterns_dict', 'rb'))

sent = 'there is 1 yellow item, 1 blue item and one black item'
program = 'and and exist filter ALL_ITEMS lambda_x_: is_yellow x exist filter ALL_ITEMS lambda_y_: is_blue y exist filter ALL_ITEMS lambda_z_: is_black z'
sent2 = 'each box have at least 1 blue item'
print(get_formalized_sentence(sent))
# print(get_formalized_sentence(sent2))
print(get_programs_for_sentence_by_pattern(sent2, dict))
print(len(dict))
print(get_formlized_sentence_and_decoding(sent, program, dict))
print(len(dict))