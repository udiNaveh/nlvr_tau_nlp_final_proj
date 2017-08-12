import json
from structured_rep_utils import *
import definitions

train = definitions.TRAIN_JSON


def read_data(filename):
    data = []
    with open(filename) as data_file:
        for line in data_file:
            data.append(json.loads(line))
    return data


def reorgenize_data(data):
    lines_by_index = dict()
    sentences_str = dict()
    processed_data = dict()
    for line in data:

        s_index = int(str.split(line["identifier"], "-")[0])
        if s_index not in processed_data:
            processed_data[s_index] = {"sentence" : (line["sentence"]), "images" : []}
        processed_data[s_index]["images"].append(line)
        line["sentence"] = processed_data[s_index]["sentence"]
        line["structured_rep"] = Image(line["structured_rep"])

    return processed_data

def small_poc():
    data = read_data(train)
    sr = data[0]["structured_rep"]
    print(data[0]["identifier"])
    image = Image(sr)

    "there is a box with three yellow shapes and a black circle touching the wall"
    print(exist(filter(lambda box :  exist(filter(lambda x : x.touching_wall, filter_color(Color.BLACK, box) )) and count(filter_color(Color.YELLOW, box))==3 , image.get_all_boxes())))

    return


if __name__ == "__main__":
    small_poc()
