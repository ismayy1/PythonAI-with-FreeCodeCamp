from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# print(tf.__version__)

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data

# print(dftrain.head())

# Separate the target variable from the training and evaluation datasets.
y_train = dftrain.pop('survived')   # takes the survived column and removes it from the dframe, stores itn 'y_train'
y_eval = dfeval.pop('survived')


CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Create feature columns
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))



def make_input_fn(data_df, label_df, num_epochs=15, shuffle=True, batch_size=32):
    def input_function():  # inner function, this will be returned
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its labels
        if shuffle:
            ds = ds.shuffle(1000)  # randomize order of data
        ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
        return ds  # return a batch of the dataset
    return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

Linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

Linear_est.train(train_input_fn)  # train
result = Linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

# # clear_output()  # clears console output
# print(result['accuracy'])  # the result variable is simply a dict of stats about our model
# print(result)

# result = list(Linear_est.predict(eval_input_fn))
# print(result[0]['probabilities'][1])    # probabilities of survived -> 1, not survived is index -> 0


result = list(Linear_est.predict(eval_input_fn))
print(dfeval.loc[4])
print(y_eval.loc[4])
print(result[4]['probabilities'][1]) 