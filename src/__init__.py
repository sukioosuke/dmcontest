#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from src import test, load_data


def training():
    matrix = pd.merge(load_data.load_user_feature(), load_data.load_user_labels(), on=['user_id'])
    train_labels = matrix['label'].values
    train_data = matrix.drop(['user_id', 'label'], axis=1).values
    print(train_labels, len(train_labels))

    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(87,)),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=5)

    test = load_data.load_test_feture()
    result = model.predict(test.drop(['user_id'], axis=1).values)
    df = pd.DataFrame(data=result,
                      columns=['labels'])
    df['user_id'] = test['user_id']
    df = df[['user_id', 'labels']]
    df.to_csv('../result/result', index=False)


if __name__ == '__main__':
    training()
