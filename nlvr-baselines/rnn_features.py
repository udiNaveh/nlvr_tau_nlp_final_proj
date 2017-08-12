# Generates feature vectors for logical images in the dataset.
import json
import math
import numpy as np
import random
import sys
import util

import tensorflow as tf

MAX_NUM_ITEMS = 8
EOS_TOK = "_EOS"
NIL_TOK = "_NIL"
UNK_TOK = "_UNK"

### feats
# Given an image, get the features for the image and sentence.
#
# Inputs:
#    objs: the dictionary objects representing examples.
#    vocab: whether or not to construct a vocab.
#    flat: whether or not to flatten the features.
#
# Outputs:
#    featurized examples, number of features, and the constructed vocabulary.
def feats(objs, vocab = False, flat = True):
  vec = [ ]
  longest_sent = 0

  # Start vocabulary as normal.
  string_to_id = { EOS_TOK : 0, NIL_TOK : 1, UNK_TOK : 2}
  id_to_string = [ EOS_TOK, NIL_TOK, UNK_TOK ]
  for line_num, obj in enumerate(objs):
    # Load the image and the sentence.
    image_data = obj["structured_rep"] 
    sentence = obj["sentence"].lower().split(" ") + [EOS_TOK]

    # Add to vocabulary if you need to construct the vocabulary.
    if not vocab:
      for word in sentence:
        if not word in string_to_id:
          string_to_id[word] = len(id_to_string)
          id_to_string.append(word)

    # Get the true/false judgment.
    judgment = -1
    if obj["label"] == "false":
      judgment = 0
    elif obj["label"] == "true":
      judgment = 1
    else:
      print(judgment) 

    #### FEATURIZER ####
    features = [ ]

    # Each box has the same amount of features.
    for box in image_data:
      box_features = [ ]
      num_items_in_box = len(box)
      num_null_items = MAX_NUM_ITEMS - num_items_in_box

      # Each item has the same amount of features. Items that do not exist will
      # be zeroed out for this vector.
      num_features = 0
      num_box_features = 0
      for item in box:
        item_features = [ ]

        # 1 if the item exists.
        existence = [ 1 ]
        item_features += existence

        # Color features: Black, Yellow, or #0099ff (blue).
        color_feature = [ 0, 0, 0 ]
        color = item["color"]
        if color == "Black":
          color_feature[0] = 1
        elif color == "Yellow":
          color_feature[1] = 1
        elif color == "#0099ff":
          color_feature[2] == 1
        else:
          print("Color " + color + " not recognized.")
        item_features += color_feature

        # Item features: triangle, square, or circle.
        shape_feature = [ 0, 0, 0 ]
        shape = item["type"]
        if shape == "triangle":
          color_feature[0] = 1
        elif shape == "circle":
          color_feature[1] = 1
        elif shape == "square":
          color_feature[2] == 1
        else:
          print("Item " + shape + " not recognized.")
        item_features += shape_feature

        # Size features: 10, 20, or 30.
        size_feature = [ 0, 0, 0 ]
        size = item["size"]
        if size == 10:
          size_feature[0] == 1
        elif size == 20:
          size_feature[1] == 1
        elif size == 30:
          size_feature[2] == 1
        else:
          print("Size " + str(size) + " not recognized.")
        item_features += size_feature

        # Wall-touching features: top, bottom; left, right
        top_touching = 0
        bottom_touching = 0
        left_touching = 0
        right_touching = 0
        y_loc = item["y_loc"]
        x_loc = item["x_loc"]
        if y_loc + size == 100:
          bottom_touching = 1
        if x_loc + size == 100:
          right_touching = 1
        if y_loc == 0:
          top_touching = 1
        if x_loc == 0:
          left_touching = 1

        wall_touching_feature = [top_touching,
                                 bottom_touching,
                                 left_touching,
                                 right_touching]

        item_features += wall_touching_feature

        # Comparison features: for every item, compare left/right/above/below.
        # Will be zeroed out for itself and nonexistent items.
        compare_features = [ ]
        for other_item in box:
          other_item_x_loc = other_item["x_loc"]
          other_item_y_loc = other_item["y_loc"]

          item_left_of_other = 0
          item_right_of_other = 0
          item_above_other = 0
          item_below_other = 0

          if x_loc > other_item_x_loc:
            item_right_of_other = 1
          elif x_loc < other_item_x_loc:
            item_left_of_other = 1

          if y_loc > other_item_y_loc:
            item_below_other = 1
          if y_loc < other_item_y_loc:
            item_below_other = 1

          compare_feature = [item_left_of_other,
                             item_right_of_other,
                             item_above_other,
                             item_below_other]
          compare_features += compare_feature

        compare_features += [ 0 ] * (4 * num_null_items)
        item_features += compare_features

        box_features.append(item_features)
        num_features = len(item_features)

      for i in range(num_null_items):
        box_features.append([ 0 ] * len(item_features))

      overall_box_features = [ ]

      num_box_features = len(overall_box_features)
      box_features_w_box = [box_features, overall_box_features]
      features.append(box_features_w_box)

    image_features = [ ]
    num_image_features = len(image_features)

    final_features = [features, image_features]

    # Flatten the objects if you want a flat feature representation.
    if flat:
      flat_features = [ ]
      for box in final_features[0]:
        for item in box[0]:
          flat_features.extend(item)
        flat_features.extend(box[1])
      flat_features.extend(final_features[1])
      num_features = len(flat_features)
      final_features = flat_features
    vec.append((final_features, sentence, judgment)) 

  # Return the list of examples, the number of possible features, and the
  # vocabulary. 
  return vec, num_features, (string_to_id, id_to_string)

### token_to_id
# Replaces tokens with IDs from vocabulary in each example, and pads to the max
# length.
#
# Inputs:
#    examples: the examples to replace.
#    tok_map: maps from input tokens to IDs.
#    max_len: maximum length of sentence.
def token_to_id(examples, tok_map, max_len):
  new_examples = [ ]
  for example in examples:
    fv = example[0]
    sent = example[1]
    judg = example[2]

    sent_len = len(sent)
    
    # Pad with NIL toks.
    sent += [NIL_TOK] * (max_len - sent_len)
    ids = [ ]
    for tok in sent:
      new_id = tok_map[UNK_TOK]
      if tok in tok_map:
        new_id = tok_map[tok]
      ids.append(new_id)
    new_examples.append(util.example(ids, sent_len, judg, fv))
  return new_examples

### get_batch
# Gets a batch for a particular size.
#
# Inputs:
#    examples: the examples to sample from.
#    batch_size: the batch size to use.
#
# Outputs:
#    tuple of sentences, labels, sentence lengths, and images in batch.
def get_batch(examples, batch_size):
  samples = random.sample(examples, batch_size)
  
  sentences = [ ]
  lengths = [ ]
  labels = [ ]
  imgs = [ ]

  for sample in samples:
    sentences.append(sample.sentence)
    labels.append(sample.label)
    lengths.append(sample.sentence_length)
    imgs.append(sample.image)

  sents = np.array(sentences)
  labs = np.array(labels)
  lens = np.array(lengths)
  images = np.array(imgs)

  return (sentences, labels, lengths, imgs)

### get_batch
# Gets a batch for the entire set of samples. 
#
# Inputs:
#    samples: the example to use.
#
# Outputs:
#    tuple of sentences, labels, sentence lengths, and images in batch.
def dev_batch(samples):
  sentences = [ ]
  lengths = [ ]
  labels = [ ]
  imgs = [ ]

  for sample in samples:
    sentences.append(sample.sentence)
    labels.append(sample.label)
    lengths.append(sample.sentence_length)
    imgs.append(sample.image)

  sents = np.array(sentences)
  labs = np.array(labels)
  lens = np.array(lengths)
  images = np.array(imgs)

  return (sentences, labels, lengths, imgs)
