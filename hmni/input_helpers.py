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

import numpy as np
import random
import gc

try:
    from .preprocess import MyVocabularyProcessor
except:
    from preprocess import MyVocabularyProcessor


class InputHelper(object):
    vocab_processor = None

    def batch_iter(
            self,
            data,
            batch_size,
            num_epochs,
            shuffle=True,
    ):

        # Generates a batch iterator for a dataset.
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):

            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = \
                    np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    # Data Preparation
    def get_datasets(
            self,
            X_train,
            y_train,
            max_document_length,
            percent_dev,
            batch_size,
    ):
        (x1_text, x2_text, y) = \
            np.asarray(X_train.iloc[:, 0].str.lower()), np.asarray(X_train.iloc[:, 1].str.lower()), np.asarray(y_train)

        # Build vocabulary
        # print('Building vocabulary')
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
        # print('Length of loaded vocabulary ={}'.format(len(vocab_processor.vocabulary_)))

        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))

        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1 * len(y_shuffled) * percent_dev // 100
        del x1
        del x2

        # TODO: This is very crude, should use cross-validation
        (x1_train, x1_dev) = (x1_shuffled[:dev_idx], x1_shuffled[dev_idx:])
        (x2_train, x2_dev) = (x2_shuffled[:dev_idx], x2_shuffled[dev_idx:])
        (y_train, y_dev) = (y_shuffled[:dev_idx], y_shuffled[dev_idx:])
        # print('Train/Dev split for data: {:d}/{:d}'.format(len(y_train), len(y_dev)))

        sum_no_of_batches = sum_no_of_batches + len(y_train) // batch_size
        train_set = (x1_train, x2_train, y_train)
        dev_set = (x1_dev, x2_dev, y_dev)
        gc.collect()
        return train_set, dev_set, vocab_processor, sum_no_of_batches

    def getTestDataSet(
            self,
            X_test,
            y_test,
            vocab,
            max_document_length,
    ):
        (x1_temp, x2_temp, y) = np.asarray(X_test.iloc[:, 0].str.lower()), np.asarray(
            X_test.iloc[:, 1].str.lower()), np.asarray(y_test)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))

        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2, y
