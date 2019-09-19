#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf

from functools import reduce
from src import utilities

def load_user_labels():
    accept = pd.DataFrame(pd.read_csv('../compete_data/train_data/train_accept_label.csv'),
                          columns=['user_id', 'label'])
    reject = pd.DataFrame(pd.read_csv('../compete_data/train_data/train_reject_label.csv'),
                          columns=['user_id', 'label'])
    reject['label'] = reject['label'].map(lambda x: 0 if x == -1 else x)
    return pd.concat([accept, reject])


# TODO 处理非连续特征
def load_user_feature():
    dfs = []
    continuous_data1 = pd.read_csv('../compete_data/train_data/final_data_1.csv').get(['user_id', 'feat_5']).fillna(0)
    continuous_data2 = pd.read_csv('../compete_data/train_data/final_data_2.csv').drop(
        ['feat_18', 'feat_54', 'feat_55'], axis=1).fillna(0)
    continuous_data3 = pd.read_csv('../compete_data/train_data/final_data_3.csv').fillna(0)
    continuous_data4 = pd.read_csv('../compete_data/train_data/final_data_4.csv').fillna(0)
    # continuous_data5 = pd.read_csv('../compete_data/train_data/final_data_5.csv').fillna(0)
    # continuous_data6 = pd.read_csv('../compete_data/train_data/final_data_6.csv').fillna(0)
    # continuous_data7 = pd.read_csv('../compete_data/train_data/final_data_7.csv').fillna(0)
    # continuous_data8 = pd.read_csv('../compete_data/train_data/final_data_8.csv').fillna(0)
    # continuous_data9 = pd.read_csv('../compete_data/train_data/final_data_9.csv').fillna(0)
    # continuous_data10 = pd.read_csv('../compete_data/train_data/final_data_10.csv').fillna(0)
    # continuous_data11 = pd.read_csv('../compete_data/train_data/final_data_11.csv').fillna(0)
    # continuous_data12 = pd.read_csv('../compete_data/train_data/final_data_12.csv').fillna(0)
    temp = continuous_data1[['feat_5']].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    continuous_data1.drop('feat_5', axis=1)
    continuous_data1['feat_5'] = temp
    dfs.append(continuous_data1)
    temp = continuous_data2['user_id']
    continuous_data2 = continuous_data2[continuous_data2.columns[1:]].apply(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    continuous_data2['user_id'] = temp
    dfs.append(continuous_data2)
    temp = continuous_data3['user_id']
    continuous_data3 = continuous_data3[continuous_data3.columns[1:]].apply(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    continuous_data3['user_id'] = temp
    dfs.append(continuous_data3)
    temp = continuous_data4['user_id']
    # continuous_data4 = continuous_data4[continuous_data4.columns[1:]].apply(
    #     lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    # continuous_data4['user_id'] = temp
    # dfs.append(continuous_data4)

    return reduce(lambda left, right: pd.merge(left, right, on=['user_id']), dfs)


def load_test_feture():
    test_user_id = pd.read_csv('../compete_data/test_data/test_stage1.csv')
    return pd.merge(load_user_feature(), test_user_id, on=['user_id'])


if __name__ == '__main__':
    test = load_test_feture()
    print(test, test.shape)
