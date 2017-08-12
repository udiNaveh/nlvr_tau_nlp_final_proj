# Gets vocabulary or vocabulary-like data from a training set.
import json
import nltk
import sys

from nltk.util import ngrams

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

sentences = [json.loads(line)["sentence"] for line in open(sys.argv[1]).readlines()]

bigrams_file = open("number_bigrams.txt", "w")
trigrams_file = open("number_trigrams.txt", "w")
fourgrams_file = open("number_fourgrams.txt", "w")
fivegrams_file = open("number_fivegrams.txt", "w")
sixgrams_file = open("number_sixgrams.txt", "w")

all_bigrams = [ ]
all_trigrams = [ ]
all_fourgrams =  [ ]
all_fivegrams = [ ]
all_sixgrams = [ ]

for sentence in sentences:
  tokenized = nltk.word_tokenize(sentence.lower())

  bigrams = ngrams(tokenized, 2)

  for bigram in bigrams:
    for i, tok in enumerate(bigram):
      if tok.isnumeric() or tok in NUMBER_MAP:
        replaced_bigram = [bigram[j] if j != i else "_" for j in range(len(bigram))]
        if not replaced_bigram in all_bigrams:
          all_bigrams.append(replaced_bigram)

  trigrams = ngrams(tokenized, 3)
  for trigram in trigrams:
    for i, tok in enumerate(trigram):
      if tok.isnumeric() or tok in NUMBER_MAP:
        replaced_trigram = [trigram[j] if j != i else "_" for j in range(len(trigram))]
        if not replaced_trigram in all_trigrams:
          all_trigrams.append(replaced_trigram)

  fourgrams = ngrams(tokenized, 4)
  for fourgram in fourgrams:
    for i, tok in enumerate(fourgram):
      if tok.isnumeric() or tok in NUMBER_MAP:
        replaced_fourgram = [fourgram[j] if j != i else "_" for j in range(len(fourgram))]
        if not replaced_fourgram in all_fourgrams:
          all_fourgrams.append(replaced_fourgram)

  if len(tokenized) > 4:        
    fivegrams = ngrams(tokenized, 5)
    for fivegram in fivegrams:
      for i, tok in enumerate(fivegram):
        if tok.isnumeric() or tok in NUMBER_MAP:
          replaced_fivegram = [fivegram[j] if j != i else "_" for j in range(len(fivegram))]
          if not replaced_fivegram in all_fivegrams:
            all_fivegrams.append(replaced_fivegram)

  if len(tokenized) > 5:        
    sixgrams = ngrams(tokenized, 6)
    for sixgram in sixgrams:
      for i, tok in enumerate(sixgram):
        if tok.isnumeric() or tok in NUMBER_MAP:
          replaced_sixgram = [sixgram[j] if j != i else "_" for j in range(len(sixgram))]
          if not replaced_sixgram in all_sixgrams:
            all_sixgrams.append(replaced_sixgram)
for bigram in all_bigrams:
  bigrams_file.write(" ".join(bigram) + "\n")
for trigram in all_trigrams:
  trigrams_file.write(" ".join(trigram) + "\n")
for fourgram in all_fourgrams:
  fourgrams_file.write(" ".join(fourgram) + "\n")
for fivegram in all_fivegrams:
  fivegrams_file.write(" ".join(fivegram) + "\n")
for sixgram in all_sixgrams:
  sixgrams_file.write(" ".join(sixgram) + "\n")
