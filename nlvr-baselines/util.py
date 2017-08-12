# Generates feature vectors for logical images in the dataset.
import json
import math
import nltk
import numpy as np
import os
import random
import sys
import tensorflow as tf

from scipy.misc import imread

MAX_NUM_ITEMS = 8
EOS_TOK = "_EOS"
NIL_TOK = "_NIL"
UNK_TOK = "_UNK"

def img_path_dict(root):
  img_paths = { }
  for directory in os.listdir(root):
    dirpath = os.path.join(root, directory)
    if os.path.isdir(dirpath):
      for filename in os.listdir(dirpath):
        if filename.endswith(".png"):
          # Image filenames in the format split-x-y-z.png, where x-y is the
          # pres id and pres pos, and z is the permutation number
          code = "-".join(filename.split("-")[1:3])
          if not code in img_paths:
            img_paths[code] = dirpath
  return img_paths

class example():
  def __init__(self, sentence, sentence_length, label, image):
    self.sentence = sentence
    self.sentence_length = sentence_length
    self.label = label
    self.image = image

def load_examples(lines, tok_to_id, max_len, img_path_dict, split_name, how_many = 6):
  examples = [ ]
  for line in lines:
    code = line["identifier"]
    sentence = nltk.word_tokenize(line["sentence"].lower()) + [EOS_TOK]

    # Pad sentence
    sent_length = len(sentence)
    if sent_length <= max_len:
      pad_length = max_len - sent_length

      sentence += [NIL_TOK] * pad_length

      id_seq = [ ]
      for tok in sentence:
        if tok in tok_to_id:
          id_seq.append(tok_to_id[tok])
        else:
          id_seq.append(tok_to_id[UNK_TOK])

      label = 0
      if line["label"] == "true":
        label = 1

      image_dir = img_path_dict[code]
      for i in range(how_many):
        filename = split_name + "-" + code + "-" + str(i) + ".png"
        full_path = os.path.join(image_dir, filename)

        if not os.path.exists(full_path):
          # First find in the next full path...
          next_path = "/".join(image_dir.split("/")[:-1]) + "/" + str(int(image_dir.split("/")[-1]) + 1)
          full_path = os.path.join(next_path, filename)
          if not os.path.exists(full_path):
            next_path = "/".join(image_dir.split("/")[:-1]) + "/" + str(int(image_dir.split("/")[-1]) - 1)
            full_path = os.path.join(next_path, filename)
            if not os.path.exists(full_path):
              print("Can't find full image path " + str(full_path))
            else:
              image_data = imread(full_path)[:,:,:-1]
              examples.append(example(id_seq, sent_length, label, image_data))
          else:
            image_data = imread(full_path)[:,:,:-1]
            examples.append(example(id_seq, sent_length, label, image_data))
        else:
          image_data = imread(full_path)[:,:,:-1]

          examples.append(example(id_seq, sent_length, label, image_data))
  return examples

def vocab(lines):
  tok_to_id = { EOS_TOK : 0, NIL_TOK : 1, UNK_TOK : 2}
  id_to_tok = [ EOS_TOK, NIL_TOK, UNK_TOK]

  max_len = 0
  for line in lines:
    sentence = line["sentence"]
    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    if len(tokenized_sentence) > max_len:
      max_len = len(tokenized_sentence)
    for word in tokenized_sentence:
      if not word in tok_to_id:
        tok_to_id[word] = len(id_to_tok)
        id_to_tok.append(word)
  max_len += 1 

  return tok_to_id, id_to_tok, max_len
