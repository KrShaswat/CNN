from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
layers = keras.layers

# Get the data: original source is here: https://www.kaggle.com/zynicide/wine-reviews/data
URL = "https://storage.googleapis.com/sara-cloud-ml/wine_data.csv"
path = tf.keras.utils.get_file(URL.split('/')[-1], URL)

# Convert the data to a Pandas data frame
data = pd.read_csv(path)

# Shuffle the data
data = data.sample(frac=1)

# Print the first 5 rows
data.head()

# Do some preprocessing to limit the # of wine varities in the dataset
data = data[pd.notnull(data['country'])]
data = data[pd.notnull(data['price'])]
data = data.drop(data.columns[0], axis=1) 

variety_threshold = 500 # Anything that occurs less than this will be removed.
value_counts = data['variety'].value_counts()
to_remove = value_counts[value_counts <= variety_threshold].index
data.replace(to_remove, np.nan, inplace=True)
data = data[pd.notnull(data['variety'])]

# Split data into train and test
train_size = int(len(data) * .8)
print ("Train size: %d" % train_size)
print ("Test size: %d" % (len(data) - train_size))

# Train features
description_train = data['description'][:train_size]
variety_train = data['variety'][:train_size]

# Train labels
labels_train = data['price'][:train_size]

# Test features
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

# Test labels
labels_test = data['price'][train_size:]

# Create a tokenizer to preprocess our text descriptions
vocab_size = 12000 # This is a hyperparameter, experiment with different values for your dataset
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(description_train) # only fit on train

# Wide feature 1: sparse bag of words (bow) vocab_size vector 
description_bow_train = tokenize.texts_to_matrix(description_train)
description_bow_test = tokenize.texts_to_matrix(description_test)





