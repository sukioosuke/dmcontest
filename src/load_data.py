#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

def read_user_proflie():
    user_profile = pd.read_csv("../compete_data/train_data/final_data_1.csv")
    print(user_profile.shape)
    print(user_profile)

def read_user_feature():
    user_fetaure = pd.read_csv("../compete_data/train_data/final_data_2.csv")
    print(user_fetaure.shape)
    print(user_fetaure)

if __name__ == '__main__':
    read_user_proflie()