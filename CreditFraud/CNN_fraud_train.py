import pickle
import tensorflow as tf
import numpy as np
import data_handle as prep
import time
import os
import datetime
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skmetric


# Graph Parameters
num_checkpoints = 5
evaluate_every = 100
checkpoint_every = 100

allow_soft_placement = True
log_device_placement = False
dev_sample_percentage = .2

# Data Load
with open('./smote_data.pkl', 'rb') as f:
    X_sm, y_sm = pickle.load(f)

# x = X_sm
x = X_sm[:, 1:]
# x = StandardScaler().fit_transform(X_sm)
# x = StandardScaler().fit_transform(X_sm[:, 1:])
y = np.array(prep.labeller(y_sm))

# Parameters
batch_size = 128
num_epochs = 10
feature_length = 29
# feature_length = 30
# embedding_size = 30
num_classes = 2
# filter_size1 = 3        # filter_shape = [row_filter_size, column_filter_size, channel_size, num_filter]
num_filters1 = 128
# pooling_size1 = 2
# filter_size2 = 3
num_filters2 = 128
# pooling_size2 = 2
dropout_keep_prob_val = 1.0

l2_reg_lambda = 0.0




# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

"""
Session Start
"""
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
session_conf = tf.ConfigProto(
    allow_soft_placement=allow_soft_placement,
    log_device_placement=log_device_placement,
    gpu_options=gpu_options)

sess = tf.InteractiveSession(config=session_conf)

## Graph

# Placeholders for input, output and dropout
input_x = tf.placeholder(tf.float32, [None, feature_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# Keeping track of l2 regularization loss (optional)
l2_loss = tf.constant(0.0)


# Create a convolution + maxpool layer for each filter size
with tf.name_scope("conv-maxpool-filter1"):
    input_x_transformed = tf.reshape(input_x, [-1, 1, 1, feature_length])

    # Convolution Layer
    filter_shape1 = [1, 1, feature_length, num_filters1]   # row-filter-size, column-filter-size, channel, num_filters
    W1 = tf.Variable(tf.truncated_normal(filter_shape1, stddev=0.1), name="W1")
    b1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]), name="b1")
    conv1 = tf.nn.conv2d(
        input_x_transformed,
        W1,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv1")
    # Apply nonlinearity
    h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu1")
    # # Maxpooling over the outputs
    # pooled1 = tf.nn.max_pool(
    #     h1,
    #     ksize=[1, pooling_size1, 1, 1],
    #     # ksize=[1, sequence_length - filter_size1 + 1, 1, 1],
    #     strides=[1, pooling_size1, 1, 1],
    #     # strides=[1, 1, 1, 1],
    #     padding='VALID',
    #     name="pool1")


with tf.name_scope("conv-maxpool-filter2"):
    # Convolution Layer
    filter_shape2 = [1, 1, num_filters1, num_filters2]   # embedding_filter x embedding x 1 x num_filters
    W2 = tf.Variable(tf.truncated_normal(filter_shape2, stddev=0.1), name="W2")
    b2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]), name="b2")
    conv2 = tf.nn.conv2d(
        h1,
        W2,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv2")
    # Apply nonlinearity
    h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu2")
    # # Maxpooling over the outputs
    # pooled2 = tf.nn.max_pool(
    #     h2,
    #     ksize=[1, pooling_size2, 1, 1],
    #     # ksize=[1, sequence_length - filter_size1 + 1, 1, 1],
    #     strides=[1, pooling_size2, 1, 1],
    #     # strides=[1, 1, 1, 1],
    #     padding='VALID',
    #     name="pool2")

# Combine all the pooled features
num_filters_total = num_filters2 * 1 * 1  # Output pixel 수
h_pool = tf.reshape(h2, [-1, num_filters_total])


# Add dropout
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool, dropout_keep_prob)

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

# # Recall / Precision / F-Measure
# predicted = predictions
# actual = tf.argmax(input_y, 1)
#
# # Count true positives, true negatives, false positives and false negatives.
# tp = tf.count_nonzero(predicted * actual)
# tn = tf.count_nonzero((predicted - 1) * (actual - 1))
# fp = tf.count_nonzero(predicted * (actual - 1))
# fn = tf.count_nonzero((predicted - 1) * actual)
#
# # Calculate accuracy, precision, recall and F1 score.
# accuracy_other = (tp + tn) / (tp + fp + fn + tn)
# precision = tp / (tp + fp)
# recall = tp / (tp + fn)
# fmeasure = (2 * precision * recall) / (precision + recall)
#
# # Add metrics to TensorBoard.
# ac_other_summary = tf.summary.scalar('Accuracy_Other', accuracy_other)
# precision_summary = tf.summary.scalar('Precision', precision)
# recall_summary = tf.summary.scalar('Recall', recall)
# fmeasure_summary = tf.summary.scalar('f-measure', fmeasure)


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
# train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged, recall_summary, precision_summary])
train_summary_dir = os.path.join(out_dir, "summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

# Dev summaries
dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
# dev_summary_op = tf.summary.merge([loss_summary, acc_summary, recall_summary, precision_summary])
dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

# Initialize all variables
sess.run(tf.global_variables_initializer())

def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      input_x: x_batch,
      input_y: y_batch,
      dropout_keep_prob: dropout_keep_prob_val
    }
    _, step, summaries, loss_n, accuracy_n = sess.run(
        # [train_op, global_step, train_summary_op, loss, accuracy, recall, precision],
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
        # [global_step, dev_summary_op, loss, accuracy, recall, precision],
        [global_step, dev_summary_op, loss, accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_n, accuracy_n))
    if writer:
        writer.add_summary(summaries, step)

# Generate batches
batches = prep.batch_iter(
    list(zip(x_train, y_train)), batch_size, num_epochs)
# Training loop. For each batch...
for batch in batches:
    x_batch, y_batch = zip(*batch)
    # x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    train_step(x_batch, y_batch)
    current_step = tf.train.global_step(sess, global_step)
    if current_step % evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, writer=dev_summary_writer)
        print("")
    if current_step % checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))