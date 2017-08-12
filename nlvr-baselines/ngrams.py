import json
import nltk
import sys

from nltk.util import ngrams

sentences = [json.loads(line)["sentence"] for line in open(sys.argv[1])]

bigrams_file = open("bigrams.txt", "w")
trigrams_file = open("trigrams.txt", "w")
fourgrams_file = open("fourgrams.txt", "w")
fivegrams_file = open("fivegrams.txt", "w")
sixgrams_file = open("sixgrams.txt", "w")

all_bigrams = [ ]
all_trigrams = [ ]
all_fourgrams =  [ ]
all_fivegrams = [ ]
all_sixgrams = [ ]

for sentence in sentences:
  tokenized = nltk.word_tokenize(sentence.lower())

  bigrams = ngrams(tokenized, 2)

  for bigram in bigrams:
    if not bigram in all_bigrams:
      all_bigrams.append(bigram)
  trigrams = ngrams(tokenized, 3)

  for trigram in trigrams:
    if not trigram in all_trigrams:
      all_trigrams.append(trigram)

  fourgrams = ngrams(tokenized, 4)

  for fourgram in fourgrams:
    if not fourgram in all_fourgrams:
      all_fourgrams.append(fourgram)

  if len(tokenized) > 4:
    fivegrams = ngrams(tokenized, 5)

    for fivegram in fivegrams:
      if not fivegram in all_fivegrams:
        all_fivegrams.append(fivegram)

  if len(tokenized) > 5:
    sixgrams = ngrams(tokenized, 6)

    for sixgram in sixgrams:
      if not sixgram in all_sixgrams:
        all_sixgrams.append(sixgram)
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
