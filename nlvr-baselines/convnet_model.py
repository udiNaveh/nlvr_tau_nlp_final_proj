import json
import numpy as np
import random
import rnn_features as ft
import tensorflow as tf
import util

from simple_cnn import conv_net
from tensorflow.python.ops import rnn_cell

# Locations of sentences + img ids + labels
train_file = "../train/train.json"
dev_file = "../dev/dev.json"
test_file = "../test/test.json"

# Locations of images
train_img_root = "../train/images/"
dev_img_root = "../dev/images/"
test_img_root = "../test/images/"

# Hyperparameters
use_image = False
use_text = True
var_init = 0.01

if not (use_image or use_text):
  print("Must use at least one of text/image as input.")
  exit()

# Text hyperparameters
word_embedding_size = 16
sent_hidden_size = 128

# To embed the image, keeping the tensor not 2d.
num_channels = 3
filter_sizes = [8, 4, 2]
strides = [4,2,2]
conv_sizes = [num_channels, 32, 32]
img_out_size = 32

# To flatten the image and put through a final layer.
img_enc_size = 256
penultimate_size = 128

# Training parameters
learning_rate = 0.001
l2_loss_coeff = 0
keep_prob = 1
batch_size = 128

# Load lines from the files.
train_lines = [json.loads(line) for line in open(train_file).readlines()]  
dev_lines = [json.loads(line) for line in open(dev_file).readlines()]  
test_lines = [json.loads(line) for line in open(test_file).readlines()]

# Get the vocabulary maps.
tok_to_id, id_to_tok, max_len = util.vocab(train_lines)

vocab_size = len(id_to_tok)

# Get the paths to the images.
train_img_paths = util.img_path_dict(train_img_root)
dev_img_paths = util.img_path_dict(dev_img_root)
test_img_paths = util.img_path_dict(test_img_root)

# Load the examples. For the image-only model we don't really need to to do
# this, but for the example class we do.
train_data = util.load_examples(train_lines,
                                tok_to_id,
                                max_len,
                                train_img_paths,
                                "train")
print("Loaded the training data")
dev_data = util.load_examples(dev_lines,
                              tok_to_id,
                              max_len,
                              dev_img_paths,
                              "dev")
print("Loaded the development data")                              
valid_data = dev_data
test_data = util.load_examples(test_lines,
                               tok_to_id,
                               max_len,
                               test_img_paths,
                               "test")
print("Loaded the testing data")

# Get the mean of the train images.
train_imgs = np.stack([ex.image for ex in train_data])
mean = train_imgs.mean(0).mean(0).mean(0) / [255., 255., 255.]

graph = tf.Graph()
with graph.as_default():
  ### Placeholders
  # gold_label: the gold label (true or false), represented as 0 (false) and 1
  #             (true), the size of the batch.
  # batch_size: the batch size.
  # keep_prob: dropout likelihoods (varies between train/test if using
  #            dropout)
  #
  # If using images:
  #     image_placeholder: image shapes are 100x400 with 3 channels, in batch.
  #     img_mean: the mean of the training images (constant).
  #
  # If using text:
  #     sentence_placeholder: placeholder for sentence vocab words.
  #     sent_len_placeholder: sentence lengths.
  gold_label_placeholder = tf.placeholder(shape = [None],
                                          dtype = tf.int32,
                                          name = "gold_label_placeholder")
  batch_size_placeholder = tf.placeholder(dtype = tf.int32,
                                          name = "batch_size")
  keep_prob_placeholder = tf.placeholder(dtype = tf.float32,
                                          name = "keep_prob")

  if use_image:
    image_placeholder = tf.placeholder(shape = [None, 100, 400, 3],
                                       dtype = tf.float32,
                                       name = "image_placeholder")
    img_mean = tf.constant(mean, 
                           dtype = tf.float32,
                           name = "img_mean_placeholder")
  if use_text:
    sentence_placeholder = tf.placeholder(shape = [None, max_len],
                                          dtype = tf.int32,
                                          name = "sentence_placeholder")
    sent_len_placeholder = tf.placeholder(shape = [None],
                                          dtype = tf.int32,
                                          name = "sent_length_placeholder")

  ### Variables
  # all_weights keeps track of the weights for normalization when computing
  # loss.
  #
  # final_layer_weights: for affine at last layer.
  # final_layer_biases: for biases at last layer.
  #
  # If using image:
  #     scaling_weights: for affine on flattened result of convnet.
  #     scaling_biases: for biases on transformed convnet results.
  #
  # If using text:
  #     word_embeddings: word embeddings.
  #     encoder_cell: RNN cell for encoding.
  mid_layer_size = 0
  if use_image:
    mid_layer_size += img_enc_size
  if use_text:
    mid_layer_size += sent_hidden_size

  mid_layer_weights = tf.Variable(tf.random_normal([mid_layer_size,
                                                    penultimate_size],
                                                   stddev = var_init),
                                    name = "mid_layer_biases")
  mid_layer_biases = tf.Variable(tf.zeros([penultimate_size]),
                                   name = "mid_layer_biases")
  final_layer_weights = tf.Variable(tf.random_normal([penultimate_size, 2],
                                                      stddev = var_init),
                                    name = "final_layer_biases")
  final_layer_biases = tf.Variable(tf.zeros([2]),
                                   name = "final_layer_biases")
  all_weights = [mid_layer_weights, final_layer_weights]

  if use_image:
    img_code_size = 3*10*32 # Must modify if changing filter/stride/etc.

    scaling_weights = tf.Variable(tf.random_normal([img_code_size,
                                                    img_enc_size],
                                                   stddev = var_init),
                                  name = "image_scaling_weights")
    scaling_biases = tf.Variable(tf.zeros([img_enc_size]),
                                 name = "image_scaling_baises")
    all_weights.append(scaling_weights)
  
  if use_text:
    word_embeddings = tf.Variable(tf.random_uniform(shape = [vocab_size,
                                                             word_embedding_size],
                                                    minval = -var_init,
                                                    maxval =  var_init),
                                  name = "word_embeddings")
    all_weights.append(word_embeddings)

    encoder_cell = rnn_cell.BasicLSTMCell(sent_hidden_size,
                                          state_is_tuple = True)

  ### Computation
  # Normalize the image by putting values between 0 and 1 and subtracting the
  # mean from the training data. Also resize so the images are smaller (40x160)
  inputs_to_concat = [ ]

  if use_image:
    normalized_img = tf.div(image_placeholder, 256.) - img_mean
    image_resized = tf.image.resize_images(normalized_img, 40, 160)

    # Run image through convnet.
    img_code, weights = conv_net(image_resized,
                                 filter_sizes, 
                                 conv_sizes,
                                 strides,
                                 img_out_size)
    all_weights.extend(weights)
   
    # Flatten the image so it is no longer three channels.
    flattened_image = tf.reshape(img_code,
                                 [batch_size_placeholder, img_code_size])

    # Scale the size of the image down using an affine+biases.
    img_scaled_down = tf.matmul(flattened_image,
                                scaling_weights) + scaling_biases
    inputs_to_concat.append(img_scaled_down)
 
  # Embed the words, and run through dynamic RNN.
  if use_text:
    embedded_sentence = tf.nn.embedding_lookup(word_embeddings,
                                               sentence_placeholder)
    hidden_states, final_output = tf.nn.dynamic_rnn(cell = encoder_cell,
                                                    inputs = embedded_sentence,
                                                    sequence_length = sent_len_placeholder,
                                                    dtype = tf.float32)
    inputs_to_concat.append(final_output[0])

  # Final layer.
  mid_layer = tf.nn.relu(tf.matmul(tf.concat(1, inputs_to_concat),
                                   mid_layer_weights) + mid_layer_biases)
  
  unnormalized_probs = tf.matmul(mid_layer, final_layer_weights) + final_layer_biases
  normalized_probs = tf.nn.softmax(unnormalized_probs)
  pred_vals = tf.cast(tf.argmax(normalized_probs, dimension = 1), tf.int32)

  one_hot_gold = tf.one_hot(gold_label_placeholder, 2)

  l2_loss = 0
  for weight in all_weights:
    l2_loss += tf.nn.l2_loss(weight)

  # Loss: negative log likelihood
  loss = tf.reduce_mean(-tf.log(tf.reduce_sum(tf.mul(normalized_probs,
                                                     one_hot_gold),
                                              1))) + l2_loss_coeff * l2_loss

  accuracy = tf.reduce_mean(tf.cast(tf.equal(gold_label_placeholder,
                                             pred_vals), tf.float32))

  loss_summary = tf.scalar_summary("loss", loss)
  acc_summary = tf.scalar_summary("acc", accuracy)
  summaries = tf.merge_summary([loss_summary, acc_summary])

  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

### Execution
with tf.Session(graph = graph) as session:
  session.run(tf.initialize_all_variables())
  train_writer = tf.train.SummaryWriter("performance/train")
  dev_writer = tf.train.SummaryWriter("performance/dev")
  valid_writer = tf.train.SummaryWriter("performance/valid")

  step_num = 0

  keep_training = True
  max_acc = 0

  steps_per_epoch = int(len(train_data) / batch_size) + 1
  patience = 5
  countdown = patience

  while keep_training:
    # Run a training step.
    train_batch = ft.get_batch(train_data, batch_size)
    feed_dict = { gold_label_placeholder : train_batch[1],
                  keep_prob_placeholder : keep_prob,
                  batch_size_placeholder : batch_size }
    if use_image:
      feed_dict[image_placeholder.name] = train_batch[3]
    if use_text:
      feed_dict[sentence_placeholder.name] = train_batch[0]
      feed_dict[sent_len_placeholder.name] = train_batch[2]

    step_acc, step_loss, step_summ, _ = session.run([accuracy,
                                                     loss,
                                                     summaries,
                                                     optimizer],
                                                    feed_dict =feed_dict)
    train_writer.add_summary(step_summ, step_num)

    # If you have finished an epoch, run a validation step.
    better = False
    if step_num % steps_per_epoch  == 0 and step_num > 0:
      valid_acc = 0
      random.shuffle(valid_data)

      # Randomly split the validation data and put it in batches.
      for i in range(int(len(valid_data) / batch_size + 1)):
        partial_valid = valid_data[i * batch_size:min(len(valid_data),
                                                      (i + 1) * batch_size)]
        partial_valid_set = ft.dev_batch(partial_valid)
        feed_dict = { gold_label_placeholder : partial_valid_set[1],
                      keep_prob_placeholder: 1.,
                      batch_size_placeholder : len(partial_valid) }
        if use_image:
          feed_dict[image_placeholder.name] = partial_valid_set[3]
        if use_text:
          feed_dict[sentence_placeholder.name] = partial_valid_set[0]
          feed_dict[sent_len_placeholder.name] = partial_valid_set[2]

        partial_valid_acc, step_loss, step_summ, preds = session.run([accuracy,
                                                                      loss,
                                                                      summaries,
                                                                      pred_vals],
                                                                     feed_dict = feed_dict)
        valid_acc += partial_valid_acc * len(partial_valid)

      # Get final validation accuracy over all subsets.
      num_correct = int(valid_acc)
      valid_acc /= len(valid_data)
      countdown -= 1
      print("(" + str(countdown) + ") V: " + str(valid_acc))

      # Update patience if accuracy has improved.
      if num_correct > max_acc:
        max_acc = num_correct 
        better = True

        patience = patience * 1.1
        countdown = patience
        print("New patience: " + str(patience))

      # If countdown runs out, terminate training.
      if countdown <= 0:
        keep_training = False

      valid_writer.add_summary(step_summ, step_num)

    # If last validation step improved, run on all three other sets (dev,
    # both testing sets). Also run on the full training set.
    if better:
      # Development set
      dev_acc = 0
      random.shuffle(dev_data)
      for i in range(int(len(dev_data) / batch_size + 1)):
        partial_dev = dev_data[i * batch_size:min(len(dev_data),
                                                  (i + 1) * batch_size)]
        partial_dev_set = ft.dev_batch(partial_dev)
        feed_dict = { gold_label_placeholder : partial_dev_set[1],
                      keep_prob_placeholder: 1.,
                      batch_size_placeholder : len(partial_dev) }
        if use_image:
          feed_dict[image_placeholder.name] = partial_dev_set[3]
        if use_text:
          feed_dict[sentence_placeholder.name] = partial_dev_set[0]
          feed_dict[sent_len_placeholder.name] = partial_dev_set[2]

        partial_dev_acc, step_loss, step_summ, preds = session.run([accuracy,
                                                                    loss,
                                                                    summaries,
                                                                    pred_vals],
                                                                   feed_dict = feed_dict)
        dev_acc += partial_dev_acc * len(partial_dev)

      dev_acc /= len(dev_data)
      print("D: "  + str(dev_acc))
      dev_writer.add_summary(step_summ, step_num)

      # Public test set
      test_acc = 0
      random.shuffle(test_data)
      for i in range(int(len(test_data) / batch_size + 1)):
        partial_test = test_data[i * batch_size:min(len(test_data),
                                                   (i + 1) * batch_size)]
        partial_test_set = ft.dev_batch(partial_test)
        feed_dict = { gold_label_placeholder : partial_test_set[1],
                      keep_prob_placeholder: 1.,
                      batch_size_placeholder : len(partial_test) }
        if use_image:
          feed_dict[image_placeholder.name] = partial_test_set[3]
        if use_text:
          feed_dict[sentence_placeholder.name] = partial_test_set[0]
          feed_dict[sent_len_placeholder.name] = partial_test_set[2]

        partial_test_acc = session.run(accuracy, feed_dict = feed_dict)
        test_acc += partial_test_acc * len(partial_test)
      test_acc /= len(test_data)

      # Entire training set.
      full_train_acc = 0
      random.shuffle(train_data)
      for i in range(int(len(train_data) / batch_size + 1)):
        partial_train = train_data[i * batch_size:min(len(train_data),
                                                    (i + 1) * batch_size)]
        partial_train_set = ft.dev_batch(partial_train)
        feed_dict = {gold_label_placeholder : partial_train_set[1],
                     keep_prob_placeholder: 1.,
                     batch_size_placeholder : len(partial_train) }
        if use_image:
          feed_dict[image_placeholder.name] = partial_train_set[3]
        if use_text:
          feed_dict[sentence_placeholder.name] = partial_train_set[0]
          feed_dict[sent_len_placeholder.name] = partial_train_set[2]

        partial_train_acc = session.run(accuracy, feed_dict = feed_dict)
        full_train_acc += partial_train_acc * len(partial_train)
      full_train_acc /= len(train_data)

    step_num += 1

  print("Dev: " + str(dev_acc))
  print("Test: " + str(test_acc))
  print("Train: " + str(full_train_acc))

