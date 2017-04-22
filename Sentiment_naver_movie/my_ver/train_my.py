import pickle
import data_handle_my as prep
import numpy as np
import tensorflow as tf
import math
import time
import os
import datetime

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("filter_size1", 5, "Filter sizes for conv-layer1")
tf.flags.DEFINE_integer("filter_size2", 4, "Filter sizes for conv-layer2")
tf.flags.DEFINE_integer("filter_size3", 3, "Filter sizes for conv-layer3")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")   # success : 64
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")   # success : 200
# tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
"""
1. Loading data
2. Padding sentences
3. One-hot-encoding labels
"""
print("Loading data...")

# pos_tagger = Twitter()
#
# start_time = time.time()
# train_docs = [prep.tokenize(pos_tagger, row[1]) for row in train_data]
# train_labels = [float(row[2]) for row in train_data]
# print('---- %s seconds elapsed ----' % (time.time() - start_time))
#
# # Saving the objects:
# with open('train.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([train_docs, train_labels], f)

with open('train.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    train_docs, train_labels = pickle.load(f)

train_labels = prep.labeller(train_labels)

train_docs_p = prep.pad_sentences(train_docs)

# Build vocabulary
voc, voc_inv = prep.build_vocab(train_docs_p)
undefined_idx = voc['#UNDEFINED']
x, y = prep.build_input_data(train_docs_p, train_labels, voc)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

"""
Session Start
"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement,
    gpu_options=gpu_options)

sess = tf.Session(config=session_conf)

sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]
vocab_size = len(voc)
embedding_size = FLAGS.embedding_dim
filter_size1 = FLAGS.filter_size1
filter_size2 = FLAGS.filter_size2
filter_size3 = FLAGS.filter_size3
num_filters = FLAGS.num_filters
l2_reg_lambda = FLAGS.l2_reg_lambda
pooling_size = 3

# Placeholders for input, output and dropout
input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Keeping track of l2 regularization loss (optional)
l2_loss = tf.constant(0.0)

# Embedding layer
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W_embed = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        name="W")   # vocab_size x embedding_size
    embedded_chars = tf.nn.embedding_lookup(W_embed, input_x)   # input_x's length(95) x embedding
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)    # Add dim. to last(for channel) : 95 x embedding x 1

# Create a convolution + maxpool layer for each filter size

pooled_outputs = []
with tf.name_scope("conv-maxpool-filter1"):
    # Convolution Layer
    filter_shape1 = [filter_size1, embedding_size, 1, num_filters]   # embedding_filter x embedding x 1 x num_filters
    W1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
    conv1 = tf.nn.conv2d(
        embedded_chars_expanded,
        W1,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    # Apply nonlinearity
    h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
    # Maxpooling over the outputs
    pooled1 = tf.nn.max_pool(
        h1,
        ksize=[1, (math.floor((sequence_length - filter_size1 + 1) / pooling_size)), 1, 1],
        # ksize=[1, sequence_length - filter_size1 + 1, 1, 1],
        strides=[1, (math.floor((sequence_length - filter_size1 + 1) / pooling_size)), 1, 1],
        # strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
    pooled_outputs.append(pooled1)

with tf.name_scope("conv-maxpool-filter2"):
    # Convolution Layer
    filter_shape2 = [filter_size2, embedding_size, 1, num_filters]   # embedding_filter x embedding x 1 x num_filters
    W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W2")
    b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
    conv2 = tf.nn.conv2d(
        embedded_chars_expanded,
        W2,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    # Apply nonlinearity
    h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
    # Maxpooling over the outputs
    pooled2 = tf.nn.max_pool(
        h2,
        ksize=[1, (math.floor((sequence_length - filter_size2 + 1) / pooling_size)), 1, 1],
        # ksize=[1, sequence_length - filter_size1 + 1, 1, 1],
        strides=[1, (math.floor((sequence_length - filter_size2 + 1) / pooling_size)), 1, 1],
        # strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
    pooled_outputs.append(pooled2)

with tf.name_scope("conv-maxpool-filter3"):
    # Convolution Layer
    filter_shape3 = [filter_size3, embedding_size, 1, num_filters]   # embedding_filter x embedding x 1 x num_filters
    W3 = tf.Variable(tf.truncated_normal(filter_shape3, stddev=0.1), name="W3")
    b3 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b3")
    conv3 = tf.nn.conv2d(
        embedded_chars_expanded,
        W3,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    # Apply nonlinearity
    h3 = tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu3")
    # Maxpooling over the outputs
    pooled3 = tf.nn.max_pool(
        h3,
        ksize=[1, (math.floor((sequence_length - filter_size3 + 1) / pooling_size)), 1, 1],
        # ksize=[1, sequence_length - filter_size1 + 1, 1, 1],
        strides=[1, (math.floor((sequence_length - filter_size3 + 1) / pooling_size)), 1, 1],
        # strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
    pooled_outputs.append(pooled3)

# Combine all the pooled features
num_filters_total = num_filters * pooling_size * 3
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# Add dropout
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# Final (unnormalized) scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")

# CalculateMean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

# Accuracy
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# Define Training procedure
global_step = tf.Variable(0, name="global_step", trainable=False)
optimizer = tf.train.AdamOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Keep track of gradient values and sparsity (optional)
grad_summaries = []
for g, v in grads_and_vars:
    if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)

# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

# Summaries for loss and accuracy
loss_summary = tf.summary.scalar("loss", loss)
acc_summary = tf.summary.scalar("accuracy", accuracy)

# Train Summaries
train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# Dev summaries
dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

# Write vocabulary
# vocab_processor.save(os.path.join(out_dir, "vocab"))
# Saving the objects:
with open('vocab.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([voc, voc_inv], f)

# Initialize all variables
sess.run(tf.global_variables_initializer())

def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
      dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, summaries, loss_n, accuracy_n = sess.run(
        [train_op, global_step, train_summary_op, loss, accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_n, accuracy_n))
    train_summary_writer.add_summary(summaries, step)

def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
      dropout_keep_prob: 1.0
    }
    step, summaries, loss_n, accuracy_n = sess.run(
        [global_step, dev_summary_op, loss, accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_n, accuracy_n))
    if writer:
        writer.add_summary(summaries, step)

# Generate batches
batches = prep.batch_iter(
    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
# Training loop. For each batch...
for batch in batches:
    x_batch, y_batch = zip(*batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
    if current_step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))