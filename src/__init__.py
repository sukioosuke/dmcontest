#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

from src import test, load_data, utilities


def training():
    matrix = pd.merge(load_data.load_user_feature(), load_data.load_user_labels(), on=['user_id'])
    train_labels = matrix['label'].values
    train_data = matrix.drop(['user_id', 'label'], axis=1).values

    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(512, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
    process = model.fit(train_data, train_labels, epochs=10, validation_split=0.1, shuffle=True,
                        callbacks=[utilities.save_checkpoint('../result/chekpoint')])
    print_history(process)
    model.save('../result/model.h5')


def predict():
    test = load_data.load_test_feture()
    model = keras.models.load_model('../result/model.h5')
    result = model.predict(test.drop(['user_id'], axis=1).values)
    df = pd.DataFrame(data=result,
                      columns=['labels'])
    df['user_id'] = test['user_id']
    df = df[['user_id', 'labels']]
    df.to_csv('../result/result', header=False, index=False)


def print_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epoch times')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../result/loss.png')

    plt.clf()
    plt.plot(epochs, acc, 'bo', label='Training Acc')
    plt.plot(epochs, val_acc, 'b', label='Validation Acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epoch times')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../result/acc.png')


if __name__ == '__main__':
    training()
    predict()
