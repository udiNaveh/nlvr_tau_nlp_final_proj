import os
from preprocessing import load_dict_from_txt,  replace_words_by_dictionary, preprocess_sentences
import definitions
from general_utils import increment_count
from handle_data import read_data


formalization_file = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'formalized words.txt')
formalization_file_2 = os.path.join(definitions.DATA_DIR, 'sentence-processing', 'more_replacements.txt')

def get_sentences_formalized(sentences):
    dict = load_dict_from_txt(formalization_file)
    for i in range(2,10):
        dict[str(i)] = 'T_INT'
    dict["1"] = 'T_ONE'
    dict["one"] = 'T_ONE'
    formalized_sentences =  replace_words_by_dictionary(sentences, dict)
    return formalized_sentences


def print_unique_sents_with_counts(sentences):
    unique = {}
    for s in sentences.values():
        increment_count(unique, s)
    for s, count in sorted(unique.items(), key= lambda kvp : kvp[1], reverse=True):
        print(s,count)


if __name__ == '__main__':

    data = read_data(definitions.TRAIN_JSON)
    sentences = {}
    for line in data:
        s_index = int(str.split(line["identifier"], "-")[0])
        if s_index not in sentences:
            sentences[s_index] = line["sentence"]

    sentences = get_sentences_formalized(preprocess_sentences(sentences, mode=None, processing_type='deep'))
    #print_unique_sents_with_counts(sentences)


    # do further formalization:
    dict = load_dict_from_txt(formalization_file_2)
    formalized_sentences_hardcore = replace_words_by_dictionary(sentences, dict)
    formalized_sentences_hardcore = replace_words_by_dictionary(formalized_sentences_hardcore, dict)
    print_unique_sents_with_counts(formalized_sentences_hardcore)


