# -*- coding: utf-8 -*-
# Tensorflow Ver. 1.1
import tensorflow as tf

class Seq2Seq_basic:
    """
    Seq2Seq for one hot encoded text list that size of vocabulary size
    input_dim = vocabulary size
    """

    def __init__(self, input_dim, enc_n_hidden=64, dec_n_hidden=64, enc_n_layers=3, dec_n_layers=3):
        self.seed = 777
        self.enc_n_hidden = enc_n_hidden
        self.dec_n_hidden = dec_n_hidden
        self.enc_n_layers = enc_n_layers
        self.dec_n_layers = dec_n_layers
        self.input_dim = self.vocab_size = input_dim
        # self.enc_seq_length = 0    # Seq length list for dynamic RNN
        # self.dec_seq_length = 0  # Seq length list for dynamic RNN

        self.output_keep_prob = 1.0
        self.learning_rate = 0.001

        self.enc_input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.input_dim], name='encoder_input')
        self.dec_input = tf.placeholder(dtype=tf.float32, shape=[None, None, self.input_dim], name='decoder_input')
        self.dec_target = tf.placeholder(dtype=tf.int64, shape=[None, None], name='decoder_target')
        #
        self.enc_seq_length = tf.expand_dims(tf.shape(self.enc_input)[1], axis=0)
        self.dec_seq_length = tf.expand_dims(tf.shape(self.dec_input)[1], axis=0)

        # self.enc_seq_length = tf.placeholder(dtype=tf.int32, shape=[None,], name='encoder_seq_length')
        # self.dec_seq_length = tf.placeholder(dtype=tf.int32, shape=[None,], name='decoder_seq_length')

        self.weights = tf.Variable(tf.ones([self.dec_n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.build_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        # Building Encoder
        with tf.variable_scope('encode'):
            enc_output, enc_state = self.build_encoder(self.enc_input, self.enc_n_hidden,
                                                       self.enc_n_layers, self.output_keep_prob)

        # Building Decoder
        with tf.variable_scope('decode'):
            dec_output, dec_state = self.build_decoder(self.dec_input, self.dec_seq_length,
                                                       self.dec_n_hidden, self.dec_n_layers,
                                                       enc_state, self.output_keep_prob)

        # Building ops
        self.logits, self.cost, self.train_op = self.build_ops(dec_output, self.dec_target)

        self.outputs = tf.argmax(self.logits, 2)

    def cell(self, n_hidden, output_keep_prob):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def build_cells(self, n_hidden, n_layers, output_keep_prob):
        multi_cell = tf.contrib.rnn.MultiRNNCell([self.cell(n_hidden, output_keep_prob) for _ in range(n_layers)])
        return multi_cell

    def build_encoder(self, encoder_input, n_hidden, n_layers, output_keep_prob):
        enc_cell = self.build_cells(n_hidden, n_layers, output_keep_prob)
        enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, encoder_input,
                                                  # sequence_length=self.enc_seq_length,
                                                  time_major=False,
                                                  dtype=tf.float32)
        return enc_output, enc_state

    def build_decoder(self, decoder_input, dec_seq_length, n_hidden, n_layers, initial_state, output_keep_prob):
        dec_cell = self.build_cells(n_hidden, n_layers, output_keep_prob)
        dec_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, dec_seq_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                  helper=dec_helper,
                                                  initial_state=initial_state,
                                                  output_layer=None)
        outputs, final_state = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False)
        return outputs, final_state

    def build_ops(self, outputs, targets):
        # TODO: 수정 필요
        time_steps = tf.shape(outputs[0])[1]
        outputs = tf.reshape(outputs[0], [-1, self.dec_n_hidden])

        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)

        tf.summary.scalar('cost', cost)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.dec_target: targets})

    def test(self, session, enc_input, dec_input, targets):
        prediction_check = tf.equal(self.outputs, self.dec_target)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.dec_target, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.dec_target: targets})

    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})

    def write_logs(self, session, writer, enc_input, dec_input, targets):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.dec_target: targets})

        writer.add_summary(summary, self.global_step.eval())


class Seq2Seq_text_embedding:
    """
    Seq2Seq for one hot encoded text that size of vocabulary size
    input_dim = vocabulary_size
    embedding_dim = embedding_size
    """
    def __init__(self, input_dim, embedding_dim, enc_n_hidden=64, dec_n_hidden=64, enc_n_layers=3, dec_n_layers=3):
        self.seed = 777
        self.enc_n_hidden = enc_n_hidden
        self.dec_n_hidden = dec_n_hidden
        self.enc_n_layers = enc_n_layers
        self.dec_n_layers = dec_n_layers
        self.input_dim = self.vocab_size = input_dim
        self.embedding_dim = embedding_dim
        # self.enc_seq_length = 0    # Seq length list for dynamic RNN
        # self.dec_seq_length = 0  # Seq length list for dynamic RNN

        self.output_keep_prob = 1.0
        self.learning_rate = 0.001

        self.enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                        name='encoder_input')
        self.dec_input = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                        name='decoder_input')
        self.dec_target = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_target')
        #
        self.enc_seq_length = tf.expand_dims(tf.shape(self.enc_input)[1], axis=0)
        self.dec_seq_length = tf.expand_dims(tf.shape(self.dec_input)[1], axis=0)

        # self.enc_seq_length = tf.placeholder(dtype=tf.int32, shape=[None,], name='encoder_seq_length')
        # self.dec_seq_length = tf.placeholder(dtype=tf.int32, shape=[None,], name='decoder_seq_length')

        self.weights = tf.Variable(tf.ones([self.dec_n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0),
                name="W")
            self.enc_input_embedded = tf.nn.embedding_lookup(self.W, self.enc_input)
            self.dec_input_embedded = tf.nn.embedding_lookup(self.W, self.dec_input)

        self.build_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        # Building Encoder
        with tf.variable_scope('encode'):
            enc_output, enc_state = self.build_encoder(self.enc_input_embedded, self.enc_n_hidden,
                                                       self.enc_n_layers, self.output_keep_prob)

        # Building Decoder
        with tf.variable_scope('decode'):
            dec_output, dec_state = self.build_decoder(self.dec_input_embedded, self.dec_seq_length,
                                                       self.dec_n_hidden, self.dec_n_layers, enc_state,
                                                       self.output_keep_prob)

        # Building ops
        self.logits, self.cost, self.train_op = self.build_ops(dec_output, self.dec_target)

        self.outputs = tf.argmax(self.logits, 2)

    def cell(self, n_hidden, output_keep_prob):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def build_cells(self, n_hidden, n_layers, output_keep_prob):
        multi_cell = tf.contrib.rnn.MultiRNNCell(
            [self.cell(n_hidden, output_keep_prob) for _ in range(n_layers)])
        return multi_cell

    def build_encoder(self, encoder_input, n_hidden, n_layers, output_keep_prob):
        enc_cell = self.build_cells(n_hidden, n_layers, output_keep_prob)
        enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, encoder_input,
                                                  # sequence_length=self.enc_seq_length,
                                                  time_major=False,
                                                  dtype=tf.float32)
        return enc_output, enc_state

    def build_decoder(self, decoder_input, dec_seq_length, n_hidden, n_layers, initial_state, output_keep_prob):
        dec_cell = self.build_cells(n_hidden, n_layers, output_keep_prob)
        # Decoder input is truth
        dec_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, dec_seq_length)
        # Decoder input is output of previous
        # dec_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_input, dec_seq_length,
        #                                                                  embedding=self.W,
        #                                                                  sampling_probability=1.0, seed=self.seed)
        decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                  helper=dec_helper,
                                                  initial_state=initial_state,
                                                  output_layer=None)
        outputs, final_state = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False)
        return outputs, final_state

    def build_ops(self, outputs, targets):
        # TODO: 수정 필요
        time_steps = tf.shape(outputs[0])[1]
        outputs = tf.reshape(outputs[0], [-1, self.dec_n_hidden])

        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        # weights = tf.ones([tf.shape(logits)[0], time_steps])
        # cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(targets=targets, logits=logits, weights=weights))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,
                                                                                     global_step=self.global_step)

        tf.summary.scalar('cost', cost)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, targets):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.dec_target: targets})

    def test(self, session, enc_input, dec_input, targets):
        prediction_check = tf.equal(self.outputs, self.dec_target)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.dec_target, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.dec_target: targets})

    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})

    def write_logs(self, session, writer, enc_input, dec_input, targets):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.dec_target: targets})

        writer.add_summary(summary, self.global_step.eval())

class Seq2Seq_text_embedding_with_prev_output:
    """
    Seq2Seq for one hot encoded text that size of vocabulary size
    input_dim = vocabulary_size
    embedding_dim = embedding_size
    """

    def __init__(self, input_dim, embedding_dim, enc_n_hidden=64, dec_n_hidden=64, enc_n_layers=3,
                 dec_n_layers=3, mode='train', sampling_prob=1.0):

        # Modeling Parameters
        self.mode = mode
        self.embedding_dim = embedding_dim
        self.max_enc_sentence_length = 200
        self.max_dec_sentence_length = 20

        # Encoder Parameters
        self.enc_n_hidden = enc_n_hidden
        self.enc_n_layers = enc_n_layers
        self.input_dim = self.vocab_size = input_dim

        # Decoder Parameters
        self.dec_n_hidden = dec_n_hidden
        self.dec_n_layers = dec_n_layers

        # Training Parameters
        self.output_keep_prob = 1.0
        self.learning_rate = 0.001
        self.seed = 777
        self.sampling_prob = sampling_prob
        # 0.0 ≤ sampling_probability ≤ 1.0
        # 0.0: no sampling => `ScheduledEmbedidngTrainingHelper` is equivalent to `TrainingHelper`
        # 1.0: always sampling => `ScheduledEmbedidngTrainingHelper` is equivalent to `GreedyEmbeddingHelper`
        # Inceasing sampling over steps => Curriculum Learning

        self.enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')
        self.enc_seq_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='encoder_seq_length')

        self.dec_target = tf.placeholder(dtype=tf.int64, shape=[None, None], name='decoder_target')

        if self.mode == 'train':
            self.dec_seq_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='decoder_seq_length')
            self.dec_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='decoder_input')

        self.weights = tf.Variable(tf.ones([self.dec_n_hidden, self.vocab_size]), name="weights")
        self.bias = tf.Variable(tf.zeros([self.vocab_size]), name="bias")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Embedding layer
        self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0), name="W")
        self.enc_input_embedded = tf.nn.embedding_lookup(self.W, self.enc_input)
        if self.mode == 'train':
            self.dec_input_embedded = tf.nn.embedding_lookup(self.W, self.dec_input)

        self.build_model()

        self.saver = tf.train.Saver(tf.global_variables())

    def build_model(self):
        # Building Encoder
        with tf.variable_scope('encode'):
            enc_output, self.enc_state = self.build_encoder(self.enc_input_embedded, self.enc_n_hidden,
                                                       self.enc_n_layers, self.output_keep_prob)

        # Building Decoder
        with tf.variable_scope('decode'):
            # dec_output, dec_state = self.build_decoder(self.dec_input_embedded, self.dec_seq_length,
            #                                            self.dec_n_hidden, self.dec_n_layers, enc_state,
            #                                            self.output_keep_prob)
            dec_output, dec_state = self.build_decoder()

        # Building ops
        self.logits, self.cost, self.train_op = self.build_ops(dec_output, self.dec_target)

        self.outputs = tf.argmax(self.logits, 2)

    def cell(self, n_hidden, output_keep_prob):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=output_keep_prob)
        return rnn_cell

    def build_cells(self, n_hidden, n_layers, output_keep_prob):
        multi_cell = tf.contrib.rnn.MultiRNNCell(
            [self.cell(n_hidden, output_keep_prob) for _ in range(n_layers)])
        return multi_cell

    def build_encoder(self, encoder_input, n_hidden, n_layers, output_keep_prob):
        enc_cell = self.build_cells(n_hidden, n_layers, output_keep_prob)
        enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, encoder_input,
                                                  sequence_length=self.enc_seq_length,
                                                  time_major=False,
                                                  dtype=tf.float32)
        return enc_output, enc_state

    def build_decoder(self):
        dec_cell = self.build_cells(self.dec_n_hidden, self.dec_n_layers, self.output_keep_prob)
        # Decoder input is truth
        # dec_helper = tf.contrib.seq2seq.TrainingHelper(decoder_input, dec_seq_length)

        if self.mode == 'train':
            max_dec_len = tf.reduce_max(self.dec_seq_length, name='max_dec_len')
            # Decoder input is output of previous
            train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(self.dec_input_embedded, self.dec_seq_length,
                                                                               embedding=self.W,
                                                                               sampling_probability=self.sampling_prob,
                                                                               seed=self.seed)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                            helper=train_helper,
                                                            initial_state=self.enc_state,
                                                            output_layer=None)

            outputs, final_state = tf.contrib.seq2seq.dynamic_decode(train_decoder, output_time_major=False,
                                                                     maximum_iterations=max_dec_len)
            return outputs, final_state
        elif self.mode == 'inference' or self.mode == 'test':
            batch_size = tf.shape(self.enc_input)[0:1]
            start_token = tf.ones(batch_size, dtype=tf.int32)

            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.W,
                                                                    start_tokens=start_token,
                                                                    end_token=2)

            infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                            helper=infer_helper,
                                                            initial_state=self.enc_state,
                                                            output_layer=None)

            outputs, final_state = tf.contrib.seq2seq.dynamic_decode(infer_decoder, output_time_major=False,
                                                                     maximum_iterations=self.max_dec_sentence_length)
            return outputs, final_state


    def build_ops(self, outputs, targets):
        # TODO: 수정 필요
        time_steps = tf.shape(outputs[0])[1]
        outputs = tf.reshape(outputs[0], [-1, self.dec_n_hidden])

        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits, [-1, time_steps, self.vocab_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
        # weights = tf.ones([tf.shape(logits)[0], time_steps])
        # cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(targets=targets, logits=logits, weights=weights))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)

        tf.summary.scalar('cost', cost)

        return logits, cost, train_op

    def train(self, session, enc_input, dec_input, targets, enc_seq_len, dec_seq_len ):
        return session.run([self.train_op, self.cost],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.dec_target: targets,
                                      self.enc_seq_length: enc_seq_len,
                                      self.dec_seq_length: dec_seq_len})

    def inference(self, session, enc_input, enc_seq_len):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.enc_seq_length: enc_seq_len})

    def test(self, session, enc_input, enc_seq_len, targets):
        prediction_check = tf.equal(self.outputs, self.dec_target)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.dec_target, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.enc_seq_length: enc_seq_len,
                                      self.dec_target: targets})

    def write_logs(self, session, writer, enc_input, dec_input, targets, enc_seq_len, dec_seq_len):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.dec_target: targets,
                                                 self.enc_seq_length: enc_seq_len,
                                                 self.dec_seq_length: dec_seq_len})

        writer.add_summary(summary, self.global_step.eval())

# def main(_):
#     model = Seq2Seq_text(100)
#
# if __name__ == "__main__":
#     tf.app.run()