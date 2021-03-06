#! /usr/bin/env python
import pickle

import tensorflow as tf
import numpy as np
import os
import time
import datetime
# import data_helpers
import prepare_konlpy_2 as prep
from konlpy.tag import Twitter
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
# tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1491097515/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
# tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    # x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    # y_test = np.argmax(y_test, axis=1)

    # Load data
    print("Loading data...")

    # test_data = prep.read_data('E:/Py_Project/nsmc/ratings_test.txt')
    # # test_data = test_data[1:1000]
    #
    # pos_tagger = Twitter()
    #
    # start_time = time.time()
    # # train_docs = [prep.tokenize(pos_tagger, row[1]) for row in train_data]
    # # train_labels = np.array([row[2] for row in train_data]).reshape(-1, 1)
    # test_docs = [prep.tokenize(pos_tagger, row[1]) for row in test_data]
    # y_test = [float(row[2]) for row in test_data]
    # # y_test = np.array([float(row[2]) for row in test_data]).reshape(-1, )
    # # y_test = np.array([row[2] for row in test_data]).reshape(-1, 1)
    # print('---- %s seconds elapsed ----' % (time.time() - start_time))
    #
    # # Saving the objects:
    # with open('test.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    #     pickle.dump([test_docs, y_test], f)

    with open('test.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
        test_docs, y_test = pickle.load(f)

    # test_docs = test_docs[1:10000]
    # y_test = y_test[1:10000]


    y_test = prep.labeller(y_test)
    y_test = np.argmax(y_test, axis=1)

    x_raw = prep.pad_sentences(test_docs)

else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary

# vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
# vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# x_test = np.array(list(vocab_processor.transform(x_raw)))

with open('vocab.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    voc, voc_inv = pickle.load(f)

# Build vocabulary
    x_test = prep.build_test_data(x_raw, voc)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = prep.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum((all_predictions == y_test)))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w', encoding='utf-8') as f:
    csv.writer(f).writerows(predictions_human_readable)