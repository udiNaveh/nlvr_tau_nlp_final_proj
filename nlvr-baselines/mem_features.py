import json
import nltk
import random
import sys

from nltk.util import ngrams

# Map natural language numbers to numerical values.
NUMBER_MAP = { "zero" : 0,
               "no" : 0,
               "a": 1,
               "one" : 1,
               "two" : 2,
               "three" : 3,
               "four" : 4,
               "five" : 5,
               "six" : 6,
               "seven" : 7,
               "eight" : 8,
               "nine" : 9,
               "ten" : 10,
               "eleven" : 11,
               "twelve" : 12,
               "thirteen" : 13,
               "fourteen" : 14,
               "fifteen" : 15,
               "sixteen" : 16 }

objs = [json.loads(line) for line in open(sys.argv[1]).readlines()]
number_bigrams = [line.strip() for line in open("ngram_data/number_bigrams.txt").readlines()]
number_trigrams = [line.strip() for line in open("ngram_data/number_trigrams.txt").readlines()]
number_fourgrams = [line.strip() for line in open("ngram_data/number_fourgrams.txt").readlines()]
number_fivegrams = [line.strip() for line in open("ngram_data/number_fivegrams.txt").readlines()]
number_sixgrams = [line.strip() for line in open("ngram_data/number_sixgrams.txt").readlines()]
bigrams = [line.strip() for line in open("ngram_data/bigrams.txt").readlines()]
trigrams = [line.strip() for line in open("ngram_data/trigrams.txt").readlines()]
fourgrams = [line.strip() for line in open("ngram_data/fourgrams.txt").readlines()]
fivegrams = [line.strip() for line in open("ngram_data/fivegrams.txt").readlines()]
sixgrams = [line.strip() for line in open("ngram_data/sixgrams.txt").readlines()]

outfile = open(sys.argv[2], "w")

for i, obj in enumerate(objs):
  image = obj["structured_rep"]
  sentence = obj["sentence"]
  final_eval = obj["label"]
  judgment = "1"
  if final_eval == "false":
    judgment = "0"

  # For each box, count the number of color-shape objects in the box.
  num_yellow_square_in_box = [ ]
  num_yellow_triangle_in_box = [ ]
  num_yellow_circle_in_box = [ ]
  num_black_square_in_box = [ ]
  num_black_triangle_in_box = [ ]
  num_black_circle_in_box = [ ]
  num_blue_square_in_box = [ ]
  num_blue_triangle_in_box = [ ]
  num_blue_circle_in_box = [ ]

  # Also count the number of color-shape objects touching the wall in each box.
  num_yellow_square_touching_in_box = [ ]
  num_yellow_triangle_touching_in_box = [ ]
  num_yellow_circle_touching_in_box = [ ]
  num_black_square_touching_in_box = [ ]
  num_black_triangle_touching_in_box = [ ]
  num_black_circle_touching_in_box = [ ]
  num_blue_square_touching_in_box = [ ]
  num_blue_triangle_touching_in_box = [ ]
  num_blue_circle_touching_in_box = [ ]

  # Count the number of objects touching the wall in each box.
  num_touching_in_box = [ ]

  # The number of boxes in which all objects are a certain color.
  num_all_black = 0
  num_all_blue = 0
  num_all_yellow = 0

  # The count of objects in each box.
  num_total_in_box = [ ]

  # Lists the colors which have the highest/lowest y-coordinates in the image.
  top_colors = [ ]
  bottom_colors = [ ]

  for box in image:
    # Keep track of the colors in the box, and save if they are all the same.
    box_color = box[0]["color"]
    same_color = True

    num_yellow_square = 0
    num_yellow_triangle = 0
    num_yellow_circle = 0
    num_black_square = 0
    num_black_triangle = 0
    num_black_circle = 0
    num_blue_square = 0
    num_blue_triangle = 0
    num_blue_circle = 0

    num_yellow_square_touching = 0
    num_yellow_triangle_touching = 0
    num_yellow_circle_touching = 0
    num_black_square_touching = 0
    num_black_triangle_touching = 0
    num_black_circle_touching = 0
    num_blue_square_touching = 0
    num_blue_triangle_touching = 0
    num_blue_circle_touching = 0

    # Highest position in the box.
    top_pos = 80

    top_color = ""
    bottom_color = ""

    for item in box:
      color = item["color"]
      shape = item["type"]
      x_pos = item["x_loc"]
      y_pos = item["y_loc"]
      size = item["size"]

      # The object is touching a wall.
      touching = False
      if y_pos + size == 100 or x_pos + size == 100 or y_pos == 0 or x_pos == 0:
        touching = True

      # Keep track if the object is higher than the other ones, or touching the
      # bottom. This feature should be for towers only, so we only keep track
      # of bottom colors when they are actually touching the bottom.
      if y_pos <= top_pos:
        top_pos = y_pos
        top_color = color
      if y_pos + size == 100:
        bottom_color = color

      if color != box_color:
        same_color = False

      # Counts of color-shapes (and also if they are touching).
      if color == "Yellow":
        if shape == "square":
          num_yellow_square += 1
          num_yellow_square_touching += int(touching)
        elif shape == "triangle":
          num_yellow_triangle += 1
          num_yellow_triangle_touching += int(touching)
        elif shape == "circle":
          num_yellow_circle += 1
          num_yellow_circle_touching += int(touching)
      elif color == "Black":
        if shape == "square":
          num_black_square += 1
          num_black_square_touching += int(touching)
        elif shape == "triangle":
          num_black_triangle += 1
          num_black_triangle_touching += int(touching)
        elif shape == "circle":
          num_black_circle += 1
          num_black_circle_touching += int(touching)
      elif color == "#0099ff":
        if shape == "square":
          num_blue_square += 1
          num_blue_square_touching += int(touching)
        elif shape == "triangle":
          num_blue_triangle += 1
          num_blue_triangle_touching += int(touching)
        elif shape == "circle":
          num_blue_circle += 1
          num_blue_circle_touching += int(touching)
    if same_color:
      if box_color == "Yellow":
        num_all_yellow += 1
      elif box_color == "Black":
        num_all_black += 1
      elif box_color == "#0099ff":
        num_all_blue += 1

    # Add top/bottom colors to sets of colors at the top/bottom in image.
    if not top_color in top_colors and top_color != "":
      top_colors.append(top_color)
    if not bottom_color in bottom_colors and bottom_color != "":
      bottom_colors.append(bottom_color)

    # Update counts in this box.
    num_yellow_square_in_box.append(num_yellow_square)
    num_yellow_circle_in_box.append(num_yellow_circle)
    num_yellow_triangle_in_box.append(num_yellow_triangle)
    num_blue_square_in_box.append(num_blue_square)
    num_blue_circle_in_box.append(num_blue_circle)
    num_blue_triangle_in_box.append(num_blue_triangle)
    num_black_square_in_box.append(num_black_square)
    num_black_circle_in_box.append(num_black_circle)
    num_black_triangle_in_box.append(num_black_triangle)

    num_yellow_square_touching_in_box.append(num_yellow_square_touching)
    num_yellow_circle_touching_in_box.append(num_yellow_circle_touching)
    num_yellow_triangle_touching_in_box.append(num_yellow_triangle_touching)
    num_blue_square_touching_in_box.append(num_blue_square_touching)
    num_blue_circle_touching_in_box.append(num_blue_circle_touching)
    num_blue_triangle_touching_in_box.append(num_blue_triangle_touching)
    num_black_square_touching_in_box.append(num_black_square_touching)
    num_black_circle_touching_in_box.append(num_black_circle_touching)
    num_black_triangle_touching_in_box.append(num_black_triangle_touching)

    num_total_in_box.append(num_yellow_square + num_yellow_circle + num_yellow_triangle + num_blue_square + num_blue_circle + num_blue_triangle + num_black_square + num_black_triangle + num_black_circle)
    num_touching_in_box.append(num_yellow_square_touching + num_yellow_circle_touching + num_yellow_triangle_touching + num_blue_square_touching + num_blue_circle_touching + num_blue_triangle_touching + num_black_square_touching + num_black_triangle_touching + num_black_circle_touching)

  # Update counts in the entire image.
  num_yellow_square_total = sum(num_yellow_square_in_box)
  num_yellow_circle_total = sum(num_yellow_circle_in_box)
  num_yellow_triangle_total = sum(num_yellow_triangle_in_box)
  num_blue_square_total = sum(num_blue_square_in_box)
  num_blue_circle_total = sum(num_blue_circle_in_box)
  num_blue_triangle_total = sum(num_blue_triangle_in_box)
  num_black_square_total = sum(num_black_square_in_box)
  num_black_circle_total = sum(num_black_circle_in_box)
  num_black_triangle_total = sum(num_black_triangle_in_box)

  num_yellow_square_touching_total = sum(num_yellow_square_touching_in_box)
  num_yellow_circle_touching_total = sum(num_yellow_circle_touching_in_box)
  num_yellow_triangle_touching_total = sum(num_yellow_triangle_touching_in_box)
  num_blue_square_touching_total = sum(num_blue_square_touching_in_box)
  num_blue_circle_touching_total = sum(num_blue_circle_touching_in_box)
  num_blue_triangle_touching_total = sum(num_blue_triangle_touching_in_box)
  num_black_square_touching_total = sum(num_black_square_touching_in_box)
  num_black_circle_touching_total = sum(num_black_circle_touching_in_box)
  num_black_triangle_touching_total = sum(num_black_triangle_touching_in_box)

  num_touching = sum(num_touching_in_box)

  ### check_features
  # This function compares the integer value and all counts accumulated above,
  # triggering a feature if the two match. The feature triggered combines the
  # name of the count and the ngram from which the integer value came (having
  # removed the number from the ngram).
  #
  # Inputs:
  #    tok_val: integer value representing the number extracted from the ngram.
  #    replaced_ngram: ngram with the number replaced by a special token.
  #
  # Outputs:
  #    a feature vector containing each feature triggered by the model.
  def check_features(tok_val, replaced_ngram):
    added_features = [ ]
    if tok_val == num_yellow_square_total:
      added_features.append("yellowsquare#" + replaced_ngram)
    if tok_val == num_yellow_circle_total:
      added_features.append("yellowcircle#" + replaced_ngram)
    if tok_val == num_yellow_triangle_total:
      added_features.append("yellowtriangle#" + replaced_ngram)
    if tok_val == num_blue_square_total:
      added_features.append("bluesquare#" + replaced_ngram)
    if tok_val == num_blue_circle_total:
      added_features.append("bluecircle#" + replaced_ngram)
    if tok_val == num_blue_triangle_total:
      added_features.append("bluetriangle#" + replaced_ngram)
    if tok_val == num_black_square_total:
      added_features.append("blacksquare#" + replaced_ngram)
    if tok_val == num_black_circle_total:
      added_features.append("blackcircle#" + replaced_ngram)
    if tok_val == num_black_triangle_total:
      added_features.append("blacktriangle#" + replaced_ngram)

    if tok_val == num_all_black:
      added_features.append("allblack#" + replaced_ngram)
    if tok_val == num_all_blue:
      added_features.append("allblue#" + replaced_ngram)
    if tok_val == num_all_yellow:
      added_features.append("allyellow#" + replaced_ngram)

    if tok_val in num_yellow_square_in_box:
      added_features.append("yellowsquarebox#" + replaced_ngram)
    if tok_val in num_yellow_circle_in_box:
      added_features.append("yellowcirclebox#" + replaced_ngram)
    if tok_val in num_yellow_triangle_in_box:
      added_features.append("yellowtrianglebox#" + replaced_ngram)
    if tok_val in num_black_square_in_box:
      added_features.append("blacksquarebox#" + replaced_ngram)
    if tok_val in num_black_circle_in_box:
      added_features.append("blackcirclebox#" + replaced_ngram)
    if tok_val in num_black_triangle_in_box:
      added_features.append("blacktrianglebox#" + replaced_ngram)
    if tok_val in num_blue_square_in_box:
      added_features.append("bluesquarebox#" + replaced_ngram)
    if tok_val in num_blue_circle_in_box:
      added_features.append("bluecirclebox#" + replaced_ngram)
    if tok_val in num_blue_triangle_in_box:
      added_features.append("bluetrianglebox#" + replaced_ngram)

    if tok_val in num_total_in_box:
      added_features.append("totalbox#" + replaced_ngram)

    if tok_val <= num_yellow_square_total:
      added_features.append("atleastyellowsquare#" + replaced_ngram)
    if tok_val <= num_yellow_circle_total:
      added_features.append("atleastyellowcircle#" + replaced_ngram)
    if tok_val <= num_yellow_triangle_total:
      added_features.append("atleastyellowtriangle#" + replaced_ngram)
    if tok_val <= num_blue_square_total:
      added_features.append("atleastbluesquare#" + replaced_ngram)
    if tok_val <= num_blue_circle_total:
      added_features.append("atleastbluecircle#" + replaced_ngram)
    if tok_val <= num_blue_triangle_total:
      added_features.append("atleastbluetriangle#" + replaced_ngram)
    if tok_val <= num_black_square_total:
      added_features.append("atleastblacksquare#" + replaced_ngram)
    if tok_val <= num_black_circle_total:
      added_features.append("atleastblackcircle#" + replaced_ngram)
    if tok_val <= num_black_triangle_total:
      added_features.append("atleastblacktriangle#" + replaced_ngram)
    if tok_val <= num_all_black:
      added_features.append("atleastallblack#" + replaced_ngram)
    if tok_val <= num_all_blue:
      added_features.append("atleastallblue#" + replaced_ngram)
    if tok_val <= num_all_yellow:
      added_features.append("atleastallyellow#" + replaced_ngram)

    if tok_val <= min(num_total_in_box):
      added_features.append("atleasttotalbox#" + replaced_ngram)

    if tok_val <= min(num_yellow_square_in_box):
      added_features.append("atleastyellowsquarebox#" + replaced_ngram)
    if tok_val <= min(num_yellow_circle_in_box):
      added_features.append("atleastyellowcirclebox#" + replaced_ngram)
    if tok_val <= min(num_yellow_triangle_in_box):
      added_features.append("atleastyellowtrianglebox#" + replaced_ngram)
    if tok_val <= min(num_black_square_in_box):
      added_features.append("atleastblacksquarebox#" + replaced_ngram)
    if tok_val <= min(num_black_circle_in_box):
      added_features.append("atleastblackcirclebox#" + replaced_ngram)
    if tok_val <= min(num_black_triangle_in_box):
      added_features.append("atleastblacktrianglebox#" + replaced_ngram)
    if tok_val <= min(num_blue_square_in_box):
      added_features.append("atleastbluesquarebox#" + replaced_ngram)
    if tok_val <= min(num_blue_circle_in_box):
      added_features.append("atleastbluecirclebox#" + replaced_ngram)
    if tok_val <= min(num_blue_triangle_in_box):
      added_features.append("atleastbluetrianglebox#" + replaced_ngram)

    if tok_val == num_yellow_square_touching_total:
      added_features.append("yellowsquaretouching#" + replaced_ngram)
    if tok_val == num_yellow_circle_touching_total:
      added_features.append("yellowcircletouching#" + replaced_ngram)
    if tok_val == num_yellow_triangle_touching_total:
      added_features.append("yellowtriangletouching#" + replaced_ngram)
    if tok_val == num_blue_square_touching_total:
      added_features.append("bluesquaretouching#" + replaced_ngram)
    if tok_val == num_blue_circle_touching_total:
      added_features.append("bluecircletouching#" + replaced_ngram)
    if tok_val == num_blue_triangle_touching_total:
      added_features.append("bluetriangletouching#" + replaced_ngram)
    if tok_val == num_black_square_touching_total:
      added_features.append("blacksquaretouching#" + replaced_ngram)
    if tok_val == num_black_circle_touching_total:
      added_features.append("blackcircletouching#" + replaced_ngram)
    if tok_val == num_black_triangle_touching_total:
      added_features.append("blacktriangletouching#" + replaced_ngram)

    if tok_val in num_yellow_square_touching_in_box:
      added_features.append("yellowsquareboxtouching#" + replaced_ngram)
    if tok_val in num_yellow_circle_touching_in_box:
      added_features.append("yellowcircleboxtouching#" + replaced_ngram)
    if tok_val in num_yellow_triangle_touching_in_box:
      added_features.append("yellowtriangleboxtouching#" + replaced_ngram)
    if tok_val in num_black_square_touching_in_box:
      added_features.append("blacksquareboxtouching#" + replaced_ngram)
    if tok_val in num_black_circle_touching_in_box:
      added_features.append("blackcircleboxtouching#" + replaced_ngram)
    if tok_val in num_black_triangle_touching_in_box:
      added_features.append("blacktriangleboxtouching#" + replaced_ngram)
    if tok_val in num_blue_square_touching_in_box:
      added_features.append("bluesquareboxtouching#" + replaced_ngram)
    if tok_val in num_blue_circle_touching_in_box:
      added_features.append("bluecircleboxtouching#" + replaced_ngram)
    if tok_val in num_blue_triangle_touching_in_box:
      added_features.append("bluetriangleboxtouching#" + replaced_ngram)

    if tok_val <= num_yellow_square_touching_total:
      added_features.append("atleastyellowsquaretouching#" + replaced_ngram)
    if tok_val <= num_yellow_circle_touching_total:
      added_features.append("atleastyellowcircletouching#" + replaced_ngram)
    if tok_val <= num_yellow_triangle_touching_total:
      added_features.append("atleastyellowtriangletouching#" + replaced_ngram)
    if tok_val <= num_blue_square_touching_total:
      added_features.append("atleastbluesquaretouching#" + replaced_ngram)
    if tok_val <= num_blue_circle_touching_total:
      added_features.append("atleastbluecircletouching#" + replaced_ngram)
    if tok_val <= num_blue_triangle_touching_total:
      added_features.append("atleastbluetriangletouching#" + replaced_ngram)
    if tok_val <= num_black_square_touching_total:
      added_features.append("atleastblacksquaretouching#" + replaced_ngram)
    if tok_val <= num_black_circle_touching_total:
      added_features.append("atleastblackcircletouching#" + replaced_ngram)
    if tok_val <= num_black_triangle_touching_total:
      added_features.append("atleastblacktriangletouching#" + replaced_ngram)

    if tok_val <= min(num_yellow_square_touching_in_box):
      added_features.append("atleastyellowsquareboxtouching#" + replaced_ngram)
    if tok_val <= min(num_yellow_circle_touching_in_box):
      added_features.append("atleastyellowcircleboxtouching#" + replaced_ngram)
    if tok_val <= min(num_yellow_triangle_touching_in_box):
      added_features.append("atleastyellowtriangleboxtouching#" + replaced_ngram)
    if tok_val <= min(num_black_square_touching_in_box):
      added_features.append("atleastblacksquareboxtouching#" + replaced_ngram)
    if tok_val <= min(num_black_circle_touching_in_box):
      added_features.append("atleastblackcircleboxtouching#" + replaced_ngram)
    if tok_val <= min(num_black_triangle_touching_in_box):
      added_features.append("atleastblacktriangleboxtouching#" + replaced_ngram)
    if tok_val <= min(num_blue_square_touching_in_box):
      added_features.append("atleastbluesquareboxtouching#" + replaced_ngram)
    if tok_val <= min(num_blue_circle_touching_in_box):
      added_features.append("atleastbluecircleboxtouching#" + replaced_ngram)
    if tok_val <= min(num_blue_triangle_touching_in_box):
      added_features.append("atleastbluetriangleboxtouching#" + replaced_ngram)
    return added_features

  ### feats_for_ngram
  # Computes set of features for ngram.
  # 
  # Inputs:
  #    ngram: the ngram to get features for.
  #    ngram_set: the set of ngrams from the training set.
  #    number_ngrams: set of ngrams from training set which were found to
  #                   include numbers in them (with number removed).
  #
  # Outputs:
  #    set of features for the ngram.
  def feats_for_ngram(ngram, ngram_set, number_ngrams):
    features = [ ]
    str_val = " ".join(ngram)
    # Check if the ngram is in plain ngrams, and trigger for each top/bottom
    # color as well as whether or not there are objects touching the walls.
    if str_val in ngram_set:
      replaced_val = str_val.replace(" ", "-")
      for top_color in top_colors:
        features.append("top-" + top_color + "/" + replaced_val)
      for bottom_color in top_colors:
        features.append("bottom-" + bottom_color + "/" + replaced_val)
      if num_touching > 0:
        features.append("touching/" + replaced_val)

    # For each token in the n-gram, 
    for i, tok in enumerate(ngram):
      if tok.isnumeric() or tok in NUMBER_MAP:
        replaced_version = " ".join([ngram[j] if j != i else "_" for j in range(len(ngram))]) 
        if replaced_version in number_ngrams:
          tok_val = -1
          if tok.isnumeric():
            tok_val = int(tok)
          else:
            tok_val = NUMBER_MAP[tok]
          replaced_ngram = replaced_version.replace(" ", "-")
          features += check_features(tok_val, replaced_ngram)
    return features

  # Tokenize the sentence.
  tokenized_sentence = nltk.word_tokenize(sentence.lower())
  features = [ ]

  # Bigrams.
  sentence_bigrams = list(ngrams(tokenized_sentence, 2))
  for bigram in sentence_bigrams:
    features += feats_for_ngram(bigram, bigrams, number_bigrams)

  # Trigrams.
  sentence_trigrams = list(ngrams(tokenized_sentence, 3))
  for trigram in sentence_trigrams:
    features += feats_for_ngram(trigram, trigrams, number_trigrams)

  # 4-grams.
  sentence_fourgrams = list(ngrams(tokenized_sentence, 4))
  for fourgram in sentence_fourgrams:
    features += feats_for_ngram(fourgram, fourgrams, number_fourgrams)

  # 5-grams
  if len(tokenized_sentence) > 4:
    sentence_fivegrams = list(ngrams(tokenized_sentence, 5))
    for fivegram in sentence_fivegrams:
      features += feats_for_ngram(fivegram, fivegrams, number_fivegrams)

  # 6-grams
  if len(tokenized_sentence) > 5:
    sentence_sixgrams = list(ngrams(tokenized_sentence, 6))
    for sixgram in sentence_sixgrams:
      features += feats_for_ngram(sixgram, sixgrams, number_sixgrams)
      
  new_features = [ ]
  for feature in features:
    if not feature in new_features:
      new_features.append(feature)
  outfile.write(judgment + " " + " ".join(new_features) + "\n")

