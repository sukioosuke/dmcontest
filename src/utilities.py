#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
import os


def save_checkpoint(path):
    checkpoint_path = path
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1, period=5)
    return cp_callback
