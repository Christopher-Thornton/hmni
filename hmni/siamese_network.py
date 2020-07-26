# tensorflow based implementation of deep siamese LSTM network.
# Taken from https://github.com/dhwajraj/deep-siamese-text-similarity as of 2020-07-20
# and modified to fit hmni prediction pipeline
# deep-siamese-text-similarity original copyright:
#
# MIT License
#
# Copyright (c) 2016 Dhwaj Raj
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf


class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """

    def BiRNN(self, x, dropout, scope, hidden_units):
        n_hidden = hidden_units
        n_layers = 3

        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))

        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope('fw' + scope):
            with tf.variable_scope('fw' + scope):
                stacked_rnn_fw = []
                for _ in range(n_layers):
                    fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_fw_cell = \
                        tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                    stacked_rnn_fw.append(lstm_fw_cell)
                lstm_fw_cell_m = \
                    tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope('bw' + scope):
            with tf.variable_scope('bw' + scope):
                stacked_rnn_bw = []
                for _ in range(n_layers):
                    bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_bw_cell = \
                        tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                    stacked_rnn_bw.append(lstm_bw_cell)
                lstm_bw_cell_m = \
                    tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)

        # Get lstm cell output
        with tf.name_scope('bw' + scope):
            with tf.variable_scope('bw' + scope):
                (outputs, _, _) = \
                    tf.nn.static_bidirectional_rnn(lstm_fw_cell_m,
                                                   lstm_bw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        tmp2 = (1 - y) * tf.square(tf.maximum(1 - d, 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2

    def __init__(self, sequence_length, vocab_size, embedding_size, hidden_units, batch_size):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name='input_x1')
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name='input_x2')
        self.input_y = tf.placeholder(tf.float32, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        # Embedding layer
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                 trainable=True, name='W')
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)

            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope('output'):
            self.out1 = self.BiRNN(
                self.embedded_chars1,
                self.dropout_keep_prob,
                'side1',
                hidden_units
            )
            self.out2 = self.BiRNN(
                self.embedded_chars2,
                self.dropout_keep_prob,
                'side2',
                hidden_units
            )
            self.distance = \
                tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keepdims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keepdims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keepdims=True))))
            self.distance = tf.reshape(self.distance, [-1], name='distance')
        with tf.name_scope('loss'):
            self.loss = self.contrastive_loss(self.input_y, self.distance, batch_size)

        # Accuracy computation is outside of this class.
        with tf.name_scope('accuracy'):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance),
                                        tf.rint(self.distance), name='temp_sim')  # auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
