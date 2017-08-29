import json
import preprocessing
import embeddings_maker
import tensorflow as tf

data = []
with open('train.json', 'r') as data_file:
    for line in data_file:
        data.append(json.loads(line))

# mapping = {1304: 'first sent', 2170: 'second sent'}

mapping = {}
for datum in data:
    s_index = int(str.split(datum["identifier"], "-")[0])
    mapping[s_index] = datum['sentence']

mapping = preprocessing.preprocess_sentences(mapping, processing_type='deep')
mapping = preprocessing.replace_rare_words_with_unk(mapping)
print(mapping)

sents = []
for mapp in mapping:
    if mapping[mapp] not in sents:
        sents.append(mapping[mapp])

for t in [1, 3, 5, 10]:
    for d in [6, 8, 12, 16]:
        name = 'embeddings_{0}iters_{1}dim'.format(t,d)
        with tf.variable_scope(name) as scope:
            embed_dict, embeds = embeddings_maker.word2vec(sents, name, iternum = t, embed_dim = d)
        print('done {0}iters {1}dim'.format(t,d))
# embeddings_maker.check_word2vec(embed_dict, embeds)

# with open('newtrain.json', 'w') as output_file:
#     for sample in data:
#         s_index = int(str.split(sample["identifier"], "-")[0])
#         if s_index in mapping:
#             sample["sentence"] = mapping[s_index]
#         json.dump(sample, output_file)
#         output_file.write('\n')
