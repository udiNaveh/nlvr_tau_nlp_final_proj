import rnn_features as ft
import json
import math
import operator
import random
import sys
import tensorflow as tf

from tensorflow.python.ops import rnn_cell

train_file = "../train/train.json"
dev_file = "../dev/dev.json"
test_file = "../test/test.json"

embedding_size = 32
sent_hidden_size = 32
sent_num_layers = 1

feat_hidden_size = 32
feat_num_layers = 1

concat_hidden_size = 32

learning_rate = 0.0075
batch_size = 128

train_lines = [json.loads(line) for line in open(train_file).readlines()]  
dev_lines = [json.loads(line) for line in open(dev_file).readlines()]  
test_lines = [json.loads(line) for line in open(test_file).readlines()]

train_examples, num_features, vocabulary = ft.feats(train_lines)

tok_to_id = vocabulary[0]
id_to_tok = vocabulary[1]
vocab_size = len(id_to_tok)

dev_examples, _, _ = ft.feats(dev_lines, True)
test_examples, _,_ = ft.feats(test_lines, True)

full_train_examples = train_examples

# Get the length of the longest item in the training set, and remove all dev
# examples which are longer.
sent_length = 0
for example in train_examples:
  ex_len = len(example[1])
  if ex_len > sent_length:
    sent_length = ex_len

new_dev = [ ]
for example in dev_examples:
  ex_len = len(example[1])
  if ex_len <= sent_length:
    new_dev.append(example)
new_test = [ ]
for example in test_examples:
  ex_len = len(example[1])
  if ex_len <= sent_length:
    new_test.append(example)
new_train = [ ]
for example in train_examples:
  ex_len = len(example[1])
  if ex_len <= sent_length:
    new_train.append(example)

# Replace tokens with IDs.
train_data = ft.token_to_id(train_examples, tok_to_id, sent_length)
dev_data = ft.token_to_id(new_dev, tok_to_id, sent_length)
test_data = ft.token_to_id(new_test, tok_to_id, sent_length)

valid_data = dev_data
full_train_data = train_data

def multilayer_perceptron(input_size, hidden_sizes, output_size, inputs, keep_prob = 1, dropout_on_last_layer = True):
  prev_layer_size = input_size
  prev_layer_output = inputs
  weights = [ ]
  biases = [ ]
  activations = [ ]

  for layer_num, layer_size in enumerate(hidden_sizes):
    # TODO: different way of initializing
    layer_weights = tf.Variable(tf.random_normal([prev_layer_size,
                                                   layer_size],
                                                   stddev = 0.01),
                                name = "weights_" + str(layer_num))
    weights.append(layer_weights)
    layer_biases = tf.Variable(tf.zeros(layer_size))
    layer = tf.nn.relu(tf.matmul(prev_layer_output, layer_weights) + layer_biases)
    layer = tf.nn.dropout(layer, keep_prob)
    activations.append(layer)

    prev_layer_size = layer_size
    prev_layer_output = layer

  final_layer_weights = tf.Variable(tf.random_normal([prev_layer_size,
                                                       output_size],stddev = 0.01),
                                    name = "final_weights")
  weights.append(final_layer_weights)
  final_layer_biases = tf.Variable(tf.zeros(output_size, name = "final_biases"))
  final_layer = tf.matmul(prev_layer_output, final_layer_weights) + final_layer_biases
  if dropout_on_last_layer:
    final_layer = tf.nn.dropout(final_layer, keep_prob)

  return final_layer, activations, weights, biases

graph = tf.Graph()
with graph.as_default():
  ### TF/model stuff
  sentence_placeholder = tf.placeholder(shape = [None, sent_length],
                                        dtype = tf.int32,
                                        name = "sentence_placeholder")
  sent_len_placeholder = tf.placeholder(shape = [None],
                                        dtype = tf.int32,
                                        name = "sent_length_placeholder")
  feature_vector_placeholder = tf.placeholder(shape = [None, num_features],
                                              dtype = tf.float32,
                                              name = "feature_vector_placeholder")
  gold_label_placeholder = tf.placeholder(shape = [None],
                                          dtype = tf.int32,
                                          name = "gold_label_placeholder")
  batch_size_placeholder = tf.placeholder(dtype = tf.int32, name = "batch_size")

  # Handle the sentence.
  #  1. word embeddings
  #  2. run through rnn
  var_unif_init = 1 / math.sqrt(vocab_size)
  word_embedding_vec = tf.Variable(tf.random_uniform(shape = [vocab_size,
                                                              embedding_size],
                                                     minval = -var_unif_init,
                                                     maxval = var_unif_init),
                                   name = "word_emb_vec")

  embedded_sent = tf.nn.embedding_lookup(word_embedding_vec,
                                         sentence_placeholder)

  encoder_cell = rnn_cell.MultiRNNCell([rnn_cell.BasicLSTMCell(sent_hidden_size,
                                                               state_is_tuple = True)] * sent_num_layers,
                                       state_is_tuple = True)

  hidden_states, final_output = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                  inputs = embedded_sent,
                                                  sequence_length = sent_len_placeholder,
                                                  dtype = tf.float32)
  encoder_final_states = tf.concat(1, [final_output[i][0] for i in range(sent_num_layers)])
  avg_outputs = tf.div(tf.reduce_sum(hidden_states, 1), tf.expand_dims(tf.cast(sent_len_placeholder, tf.float32), 1))

  # Handle the feature vector.
  #  1. run through a few layers
  embedded_feats, rnn_weights,_,_ = multilayer_perceptron(num_features,
                                            [feat_hidden_size],
                                            feat_hidden_size,
                                            feature_vector_placeholder)
  concat_inputs = tf.concat(1, [avg_outputs, embedded_feats])

  unnormalized_probs, final_weights,_,_ = multilayer_perceptron(sent_hidden_size * sent_num_layers + feat_hidden_size,
                                                [concat_hidden_size],
                                                2,
                                                concat_inputs)
  pred_vals = tf.cast(tf.argmax(unnormalized_probs, dimension = 1), tf.int32)

  normalized_probs = tf.nn.softmax(unnormalized_probs)
  one_hot_gold = tf.one_hot(gold_label_placeholder, 2)
  l2_loss = 0
  for weight in rnn_weights + final_weights:
    l2_loss += tf.nn.l2_loss(weight)
  loss = tf.reduce_mean(-tf.log(tf.reduce_sum(tf.mul(normalized_probs, one_hot_gold), 1)))

  accuracy = tf.reduce_sum(tf.cast(tf.equal(gold_label_placeholder, pred_vals),
                                   tf.float32)) / tf.cast(batch_size_placeholder,
                                                          tf.float32)
  loss_summary = tf.scalar_summary("loss", loss)
  acc_summary = tf.scalar_summary("acc", accuracy)
  summaries = tf.merge_summary([loss_summary, acc_summary])

  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

### Actually running stuff
# Train step.
with tf.Session(graph = graph) as session:
  session.run(tf.initialize_all_variables())
  train_writer = tf.train.SummaryWriter("performance/train")
  dev_writer = tf.train.SummaryWriter("performance/dev")
  valid_writer = tf.train.SummaryWriter("performance/valid")

  step_num = 0
  keep_training = True
  max_acc = 0

  steps_per_epoch = int(len(train_lines) / batch_size) + 1
  patience = 5
  countdown = patience

  while keep_training:
    train_batch = ft.get_batch(train_data, batch_size)
    feed_dict = { sentence_placeholder : train_batch[0],
                  sent_len_placeholder : train_batch[2],
                  feature_vector_placeholder : train_batch[3],
                  gold_label_placeholder : train_batch[1],
                  batch_size_placeholder : batch_size }
    step_acc, step_loss, step_summ, _ = session.run([accuracy, loss, summaries, optimizer], feed_dict = feed_dict)

    train_writer.add_summary(step_summ, step_num)

    better = False
    if step_num % steps_per_epoch  == 0 and step_num > 0:
      valid_set = ft.dev_batch(valid_data)

      feed_dict = { sentence_placeholder : valid_set[0],
                    sent_len_placeholder : valid_set[2],
                    feature_vector_placeholder : valid_set[3],
                    gold_label_placeholder : valid_set[1],
                    batch_size_placeholder : len(valid_data) }
      valid_acc, step_loss, step_summ = session.run([accuracy, loss, summaries], feed_dict = feed_dict)
      countdown -= 1
      print("(" + str(countdown) + ") V: " + str(valid_acc))

      if valid_acc > max_acc:
        max_acc = valid_acc
        better = True

        patience = patience * 1.1
        countdown = patience
        print("New patience: " + str(patience))
      if countdown <= 0:
        keep_training = False

      valid_writer.add_summary(step_summ, step_num)
    if better:
      dev_set = ft.dev_batch(dev_data)

      feed_dict = { sentence_placeholder : dev_set[0],
                    sent_len_placeholder : dev_set[2],
                    feature_vector_placeholder : dev_set[3],
                    gold_label_placeholder : dev_set[1],
                    batch_size_placeholder : len(dev_data) }
      dev_acc, step_loss, step_summ, best_preds = session.run([accuracy, loss, summaries, pred_vals], feed_dict = feed_dict)
      print("D: "  + str(dev_acc))
      dev_writer.add_summary(step_summ, step_num)
      test_set = ft.dev_batch(test_data)

      feed_dict = { sentence_placeholder : test_set[0],
                    sent_len_placeholder : test_set[2],
                    feature_vector_placeholder : test_set[3],
                    gold_label_placeholder : test_set[1],
                    batch_size_placeholder : len(test_data) }
      test_acc, step_loss, step_summ, best_preds = session.run([accuracy, loss, summaries, pred_vals], feed_dict = feed_dict)

      full_train_acc = 0
      for i in range(int(len(full_train_data) / 128 + 1)):
        partial_train = full_train_data[i * 128:min(len(full_train_data), (i + 1) * 128)]
        partial_train_set = ft.dev_batch(partial_train)
        feed_dict = {sentence_placeholder: partial_train_set[0],
                     sent_len_placeholder: partial_train_set[2],
                     feature_vector_placeholder :partial_train_set[3],
                     gold_label_placeholder : partial_train_set[1],
                     batch_size_placeholder : len(partial_train) }
        partial_train_acc = session.run(accuracy, feed_dict =feed_dict)
        full_train_acc += partial_train_acc * len(partial_train)
      full_train_acc /= len(full_train_data)

    step_num += 1

  print("Dev: " + str(dev_acc * len(dev_data) / len(dev_examples)))
  print("Test: " + str(test_acc * len(test_data) / len(test_examples)))
  print("Train: " + str(full_train_acc * len(full_train_data) / len(full_train_examples)))

