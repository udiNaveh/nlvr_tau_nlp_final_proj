import math
import operator
import random
import tensorflow as tf

### Hyperparameters
# Layer sizes
embedding_size = 32
first_layer_size = 32
second_layer_size = 32
last_layer_size = 32

# Learning parameters
learning_rate = .01
batch_size = 128
l2_coeff = 0.0

### Load the data
train_lines = open("mem_feats/train_feats.txt").readlines()
dev_lines = open("mem_feats/dev_feats.txt").readlines()
test_lines = open("mem_feats/test_feats.txt").readlines()
valid_lines = dev_lines

# Create dictionary mapping a feature name to an index.
feat_dict = { }
num_feats = 0
for line in train_lines:
  feats = line.strip().split(" ")[1:]
  for feat in feats:
    if not feat in feat_dict:
      feat_dict[feat] = num_feats

      num_feats += 1

### Model
graph = tf.Graph()
with graph.as_default():
  ### Placeholders
  # feats_placeholder: batch_size x max_features_in_batch. Treated as a
  #                    sequence of which features are triggered for each item
  #                    in the batch.
  # masks_placeholder: batch_size x max_features_in_batch. Masks for the
  #                    features which are not used for particular items in the
  #                    batch.
  # gold_labels_placeholder: the gold labels.
  # batch_size_placeholder: the batch size.
  # num_feats_placeholder: placeholder for the number of features in each batch
  #                        item (could also sum over the masks).
  feats_placeholder = tf.placeholder(shape = [None, None],
                                     dtype = tf.int32,
                                     name = "feats_placeholder")
  masks_placeholder = tf.placeholder(shape = [None, None],
                                     dtype = tf.float32,
                                     name = "masks_placeholder")
  gold_labels_placeholder = tf.placeholder(shape = [None],
                                           dtype = tf.int32,
                                           name = "gold_placeholder")
  batch_size_placeholder = tf.placeholder(dtype = tf.int32,
                                          name = "batch_placeholder")
  num_feats_placeholder = tf.placeholder(shape = [None],
                                         dtype = tf.float32,
                                         name = "num_feats")

  ### Variables
  # embedding_layer: embeddings of features
  # last_layer_weights: weights on final layer
  # last_layer_biases: biases towards yes/no answers
  init_range = 0.01
  embedding_layer = tf.Variable(tf.random_uniform([num_feats, embedding_size],
                                                  -init_range,
                                                   init_range),
                                dtype = tf.float32)

  last_layer_weights = tf.Variable(tf.random_uniform([embedding_size, 2],
                                                     -init_range,
                                                      init_range),
                                   dtype = tf.float32)
  last_layer_biases = tf.Variable(tf.zeros([2], dtype = tf.float32))

  last_layer_weights = tf.Variable(tf.random_uniform([embedding_size, 2],
                                                     -init_range,
                                                      init_range),
                                   dtype = tf.float32)
  last_layer_biases = tf.Variable(tf.zeros([2], dtype = tf.float32))

  ### Computations
  embedded_inputs = tf.div(tf.reduce_sum(tf.mul(tf.nn.embedding_lookup(embedding_layer,
                                                                       feats_placeholder),
                                                tf.expand_dims(masks_placeholder,
                                                               2)),
                                                1),
                                         tf.expand_dims(num_feats_placeholder,
                                                        1))

  value = tf.matmul(embedded_inputs, last_layer_weights) + last_layer_biases
  normalized_probs = tf.nn.softmax(value)
  pred_vals = tf.cast(tf.argmax(normalized_probs, dimension = 1), tf.int32)

  one_hot_gold = tf.one_hot(gold_labels_placeholder, 2)

  loss = tf.reduce_mean(-tf.log(tf.reduce_sum(tf.mul(normalized_probs,
                                              one_hot_gold), 1))) + l2_coeff * tf.nn.l2_loss(last_layer_weights)
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  accuracy = tf.reduce_sum(tf.cast(tf.equal(gold_labels_placeholder,
                                            pred_vals),
                                   tf.float32))/ tf.cast(batch_size_placeholder,
                                                         tf.float32)

  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  loss_summary = tf.scalar_summary("loss", loss)
  acc_summary = tf.scalar_summary("acc", accuracy)
  summaries = tf.merge_summary([loss_summary, acc_summary])

### model_inputs
# Gets a batch and prepares inputs for the placeholders.
#
# Inputs:
#    examples: set of examples to choose from.
#    batch_size: number of examples to choose. If zero, use all examples.
def model_inputs(examples, batch_size = 0):
  # Sample some examples.
  if batch_size > 0:
    samples = random.sample(examples, batch_size)
  else:
    samples = examples

  # Keep track of feature vectors, judgments, and also the largest feature
  # vector in the batch.
  fvs = [ ]
  judgments = [ ]
  largest_size = 0

  for sample in samples:
    feats = sample.strip().split(" ")[1:]
    judgment = 1
    if sample.strip().split(" ")[0] == "0":
      judgment = 0

    judgments.append(judgment)

    # Create the feature vector: look up the ID for the feature in the
    # previously constructed dictionary.
    fv = [ ]
    for feat in feats:
      if feat in feat_dict:
        fv.append(feat_dict[feat])

    if len(fv) > largest_size:
      largest_size = len(fv)

    fvs.append(fv)

  # Now pad the feature vectors to the longest in the batch, and also create
  # the masks and length vectors.
  padded_fvs = [ ]
  masks = [ ] 
  num_feats = [ ]
  for fv in fvs:
    fv_len = len(fv)
    pad_len = largest_size - fv_len
    new_fv = fv + [ 0 ] * pad_len
    padded_fvs.append(new_fv)
    masks.append([ 1 ] * fv_len + [ 0 ] * pad_len)
    num_feats.append(fv_len)

  return (padded_fvs, masks, judgments, num_feats)

### TRAINING
with tf.Session(graph = graph) as session:
  session.run(tf.initialize_all_variables())
  train_writer = tf.train.SummaryWriter("ll/train")
  dev_writer = tf.train.SummaryWriter("ll/dev")
  valid_writer = tf.train.SummaryWriter("ll/valid")

  # Stopping conditions.
  steps_per_epoch = int(len(train_lines) / batch_size) + 1
  patience = 5
  max_valid_acc = 0
  keep_training = True
  countdown = patience
  step_num = 0

  while keep_training:
    train_batch = model_inputs(train_lines, batch_size = batch_size)
    feed_dict = { feats_placeholder: train_batch[0],
                  gold_labels_placeholder: train_batch[2],
                  masks_placeholder : train_batch[1],
                  num_feats_placeholder : train_batch[3],
                  batch_size_placeholder : batch_size}

    step_acc, step_loss, step_summ,  _ = session.run([accuracy,
                                                     loss,
                                                     summaries,
                                                     optimizer],
                                                    feed_dict = feed_dict)
    train_writer.add_summary(step_summ, step_num)
    valid_is_better = False
    # Run on validation step every epoch.
    if step_num % steps_per_epoch == 0 and step_num > 0:
      valid_set = model_inputs(valid_lines)
      feed_dict = { feats_placeholder : valid_set[0],
                    gold_labels_placeholder : valid_set[2],
                    masks_placeholder : valid_set[1],
                    num_feats_placeholder : valid_set[3],
                    batch_size_placeholder : len(valid_set[0])}
      valid_acc, step_loss,  step_summ = session.run([accuracy,
                                                     loss,
                                                     summaries],
                                                    feed_dict = feed_dict)
      
      countdown -= 1
      print("(" + str(countdown) + ") V: " + str(valid_acc))

      if valid_acc > max_valid_acc:
        max_valid_acc = valid_acc
        valid_is_better = True

        patience = patience * 1.1
        countdown = patience
        print("New patience: " + str(patience))
      if countdown <= 0:
        keep_training = False

      valid_writer.add_summary(step_summ, step_num)

    # Run on development set if validation set improved. 
    if valid_is_better:
      dev_set = model_inputs(dev_lines)
      feed_dict = { feats_placeholder : dev_set[0],
                    gold_labels_placeholder : dev_set[2],
                    masks_placeholder : dev_set[1],
                    num_feats_placeholder : dev_set[3],
                    batch_size_placeholder : len(dev_set[0])}
      dev_acc, step_loss, step_summ = session.run([accuracy,
                                                   loss,
                                                   summaries],
                                                  feed_dict = feed_dict)
      print("D: " + str(dev_acc))
      dev_writer.add_summary(step_summ, step_num)
      test_set = model_inputs(test_lines)
      feed_dict = { feats_placeholder : test_set[0],
                    gold_labels_placeholder : test_set[2],
                    masks_placeholder : test_set[1],
                    num_feats_placeholder : test_set[3],
                    batch_size_placeholder : len(test_set[0])}
      test_acc = session.run([accuracy],
                                                  feed_dict = feed_dict)
      full_train_acc = 0
      for i in range(int(len(train_lines) / batch_size + 1)):
        partial_train = train_lines[i * batch_size:min(len(train_lines),
                                                      (i + 1) * batch_size)]
        partial_train_set = model_inputs(partial_train)
        feed_dict = { feats_placeholder : partial_train_set[0],
                      gold_labels_placeholder : partial_train_set[2],
                      masks_placeholder : partial_train_set[1],
                      num_feats_placeholder : partial_train_set[3],
                      batch_size_placeholder : len(partial_train_set[0])}
        acc = session.run([accuracy],
                                                    feed_dict = feed_dict)
        full_train_acc += acc[0] * len(partial_train_set[0])
      full_train_acc /= len(train_lines)
    step_num += 1

  print(dev_acc)
  print(test_acc[0])
  print(hidden_acc[0])
