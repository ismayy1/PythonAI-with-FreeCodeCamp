from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf


# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data

# print(dftrain.head())

# Separate the target variable from the training and evaluation datasets.
y_train = dftrain.pop('survived')   # takes the survived column and removes it from the dframe, stores itn 'y_train'
y_eval = dfeval.pop('survived')
# print(dftrain.head())
# print(y_train)
# print(dftrain.loc[0], y_train.loc[0])
# print(dftrain["age"])

# print(dftrain.head())

# print(dftrain.describe())

# print(dftrain.shape)

# print(y_train.head())

# print(dftrain.age.hist(bins=20))

# print(dfeval.shape)


#############################################################################

# Define categorical and numeric columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# Create feature columns
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Print feature columns
# print(feature_columns)

# print(dftrain["sex"].unique())
print(dftrain["embark_town"].unique())
