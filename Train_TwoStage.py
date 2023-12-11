import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import *
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import classification_report

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Conv1D, LSTM, Flatten, Add, Activation, \
    MaxPooling1D, Attention
import tensorflow as tf

import pickle
import seaborn as sns
from itertools import cycle
import scipy.io as scio
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb

read_number = 5
benign = pd.read_csv('RawDataset/{}.benign.csv'.format(read_number))
g_c = pd.read_csv('RawDataset/{}.gafgyt.combo.csv'.format(read_number))
g_j = pd.read_csv('RawDataset/{}.gafgyt.junk.csv'.format(read_number))
g_s = pd.read_csv('RawDataset/{}.gafgyt.scan.csv'.format(read_number))
g_t = pd.read_csv('RawDataset/{}.gafgyt.tcp.csv'.format(read_number))
g_u = pd.read_csv('RawDataset/{}.gafgyt.udp.csv'.format(read_number))
m_a = pd.read_csv('RawDataset/{}.mirai.ack.csv'.format(read_number))
m_sc = pd.read_csv('RawDataset/{}.mirai.scan.csv'.format(read_number))
m_sy = pd.read_csv('RawDataset/{}.mirai.syn.csv'.format(read_number))
m_u = pd.read_csv('RawDataset/{}.mirai.udp.csv'.format(read_number))
m_u_p = pd.read_csv('RawDataset/{}.mirai.udpplain.csv'.format(read_number))

benign['type'] = 'benign'
g_c['type'] = 'gafgyt_combo'
g_j['type'] = 'gafgyt_junk'
g_s['type'] = 'gafgyt_scan'
g_t['type'] = 'gafgyt_tcp'
g_u['type'] = 'gafgyt_udp'
m_a['type'] = 'mirai_ack'
m_sc['type'] = 'mirai_scan'
m_sy['type'] = 'mirai_syn'
m_u['type'] = 'mirai_udp'
m_u_p['type'] = 'mirai_udpplain'

data = pd.concat([g_c, g_j, g_s, g_t, g_u], axis=0, sort=False, ignore_index=True)
# data = pd.concat([m_a, m_sc, m_sy, m_u, m_u_p], axis=0, sort=False, ignore_index=True)

label = data['type']
# labels_full = data['type']
labels_full = pd.get_dummies(data['type'], prefix='type')
labels = labels_full.values

data = data.drop(columns='type')
data_st = data.copy()
data_st = data_st.values

skb = SelectKBest(f_classif, k=12)
data_new = skb.fit_transform(data_st, label)
scores = skb.scores_
p_values = skb.pvalues_
indices = np.argsort(scores)[::-1]
k_best_features = list(data.columns.values[indices[0:12]])
print('k best features are: ', k_best_features)
k_last_features = list(data.columns.values[indices[103:115]])
print('k last features are: ', k_last_features)
pickle.dump(skb, open('./model/XGBOOST-skb.pkl', 'wb'))

train_data_st = data_new
x_train, x_test, y_train, y_test = train_test_split(train_data_st, labels, test_size=0.2)

t = StandardScaler()
x_train = t.fit_transform(x_train)
x_test = t.transform(x_test)
pickle.dump(t, open('./model/XGBOOST-t.pkl', 'wb'))

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.125)

modelName = 'XGBOOST-MODEL'
clf = xgb.XGBClassifier()
clf.fit(x_train, y_train)
with open('./model/XGBOOST-MODEL.pickle', 'wb') as f:
    pickle.dump(clf, f)
