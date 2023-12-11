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
from sklearn.feature_selection import SelectKBest, f_classif

read_number = 5
benign = pd.read_csv('RawDataset/{}.benign.csv'.format(read_number))
G_c = pd.read_csv('RawDataset/{}.gafgyt.combo.csv'.format(read_number))
G_j = pd.read_csv('RawDataset/{}.gafgyt.junk.csv'.format(read_number))
G_s = pd.read_csv('RawDataset/{}.gafgyt.scan.csv'.format(read_number))
G_t = pd.read_csv('RawDataset/{}.gafgyt.tcp.csv'.format(read_number))
G_u = pd.read_csv('RawDataset/{}.gafgyt.udp.csv'.format(read_number))
M_a = pd.read_csv('RawDataset/{}.mirai.ack.csv'.format(read_number))
M_sc = pd.read_csv('RawDataset/{}.mirai.scan.csv'.format(read_number))
M_sy = pd.read_csv('RawDataset/{}.mirai.syn.csv'.format(read_number))
M_u = pd.read_csv('RawDataset/{}.mirai.udp.csv'.format(read_number))
M_up = pd.read_csv('RawDataset/{}.mirai.udpplain.csv'.format(read_number))

g_c = pd.read_csv('RawDataset/{}.gafgyt.combo.csv'.format(read_number))
g_j = pd.read_csv('RawDataset/{}.gafgyt.junk.csv'.format(read_number))
g_s = pd.read_csv('RawDataset/{}.gafgyt.scan.csv'.format(read_number))
g_t = pd.read_csv('RawDataset/{}.gafgyt.tcp.csv'.format(read_number))
g_u = pd.read_csv('RawDataset/{}.gafgyt.udp.csv'.format(read_number))
m_a = pd.read_csv('RawDataset/{}.mirai.ack.csv'.format(read_number))
m_sc = pd.read_csv('RawDataset/{}.mirai.scan.csv'.format(read_number))
m_sy = pd.read_csv('RawDataset/{}.mirai.syn.csv'.format(read_number))
m_u = pd.read_csv('RawDataset/{}.mirai.udp.csv'.format(read_number))
m_up = pd.read_csv('RawDataset/{}.mirai.udpplain.csv'.format(read_number))

benign['type'] = 'benign'

G_c['type'] = 'gafgyt'
G_j['type'] = 'gafgyt'
G_s['type'] = 'gafgyt'
G_t['type'] = 'gafgyt'
G_u['type'] = 'gafgyt'
M_a['type'] = 'mirai'
M_sc['type'] = 'mirai'
M_sy['type'] = 'mirai'
M_u['type'] = 'mirai'
M_up['type'] = 'mirai'

g_c['type'] = 'gafgyt_combo'
g_j['type'] = 'gafgyt_junk'
g_s['type'] = 'gafgyt_scan'
g_t['type'] = 'gafgyt_tcp'
g_u['type'] = 'gafgyt_udp'
m_a['type'] = 'mirai_ack'
m_sc['type'] = 'mirai_scan'
m_sy['type'] = 'mirai_syn'
m_u['type'] = 'mirai_udp'
m_up['type'] = 'mirai_udpplain'

data = pd.concat([benign, G_c, G_j, G_s, G_t, G_u, M_a, M_sc, M_sy, M_u, M_up], axis=0, sort=False, ignore_index=True)
datarf = pd.concat([benign, g_c, g_j, g_s, g_t, g_u, m_a, m_sc, m_sy, m_u, m_up], axis=0, sort=False, ignore_index=True)

labels_full = pd.get_dummies(data['type'], prefix='type')
labels = labels_full.values
labels_full_rf = pd.get_dummies(datarf['type'], prefix='type')
labels_rf = labels_full_rf.values

data = data.drop(columns='type')
data_st = data.copy()
data_st = data_st.values
datarf = datarf.drop(columns='type')
data_st_rf = datarf.copy()
data_st_rf = data_st_rf.values

train_data_st = data_st
x_train, x_test, y_train, y_test = train_test_split(train_data_st, labels, test_size=0.2)
train_data_st_rf = data_st_rf
x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(train_data_st_rf, labels_rf, test_size=0.2)

t = pickle.load(open('./model/1DCNN+BiGRU_t.pkl', 'rb'))
x_train = t.transform(x_train)
x_test = t.transform(x_test)

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.125)
x_train_cnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_validate_cnn = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], 1))

model = tf.keras.models.load_model('./model/1DCNN+BiGRU.h5')
model.summary()

y_pred = model.predict(x_test_cnn)
y_pred_cm = np.argmax(y_pred, axis=1)
y_test_cm = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_cm, y_pred_cm)
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(3, 3)
label = ['benign', 'gafgyt', 'mirai']
plt.figure(figsize=(11, 11))
sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, fmt='', cmap="Blues", vmin=0.2, cbar=True,
            annot_kws={'size': 11}, cbar_kws={'pad': 0.02})
plt.title('Confusion Matrix for ' + '1DCNN+BiGRU' + ' Model', fontsize='18')
plt.ylabel('True Class', fontsize=16)
plt.xlabel('Predicted Class', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./ans/' + '1DCNN+BiGRU' + '_CM.png', dpi=600, bbox_inches='tight')
plt.show()

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(labels.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green', 'aqua', 'darkorange', 'orange', 'fuchsia', 'lime', 'magenta'])
for i, color in zip(range(labels.shape[1]), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (Area = {1:0.2f})'.format(i + 1, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
plt.title('ROC for ' + '1DCNN+BiGRU' + ' Model', fontsize='18')
plt.ylabel('True Positive Rate', fontsize='16')
plt.xlabel('False Positive Rate', fontsize='16')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc="lower right")
plt.savefig('./ans/' + '1DCNN+BiGRU' + '_ROC.png', dpi=600, bbox_inches='tight')
plt.show()

list0 = []
list1 = []
list2 = []
test_x0 = []
test_x1 = []
test_x2 = []
test_y0 = []
test_y1 = []
test_y2 = []
y1_label = []
y2_label = []
for i in range(0, len(y_pred)):
    if y_pred[i][1] == np.max(y_pred[i], axis=0):
        if y_test_rf[i][1] == 1:
            list1.append(i)
            test_x1.append(x_test_rf[i])
            test_y1.append([1, 0, 0, 0, 0])
            # y1_label.append(1)
            # y1_label.append('gafgyt_combo')
        elif y_test_rf[i][2] == 1:
            list1.append(i)
            test_x1.append(x_test_rf[i])
            test_y1.append([0, 1, 0, 0, 0])
            # y1_label.append(2)
            # y1_label.append('gafgyt_junk')
        elif y_test_rf[i][3] == 1:
            list1.append(i)
            test_x1.append(x_test_rf[i])
            test_y1.append([0, 0, 1, 0, 0])
            # y1_label.append(3)
            # y1_label.append('gafgyt_scan')
        elif y_test_rf[i][4] == 1:
            list1.append(i)
            test_x1.append(x_test_rf[i])
            test_y1.append([0, 0, 0, 1, 0])
            # y1_label.append(4)
            # y1_label.append('gafgyt_tcp')
        elif y_test_rf[i][5] == 1:
            list1.append(i)
            test_x1.append(x_test_rf[i])
            test_y1.append([0, 0, 0, 0, 1])
            # y1_label.append(5)
            # y1_label.append('gafgyt_udp')
    elif y_pred[i][2] == np.max(y_pred[i], axis=0):
        if y_test_rf[i][6] == 1:
            list2.append(i)
            test_x2.append(x_test_rf[i])
            test_y2.append([1, 0, 0, 0, 0])
            # y2_label.append(1)
            # y2_label.append('mirai_ack')
        elif y_test_rf[i][7] == 1:
            list2.append(i)
            test_x2.append(x_test_rf[i])
            test_y2.append([0, 1, 0, 0, 0])
            # y2_label.append(2)
            # y2_label.append('mirai_scan')
        elif y_test_rf[i][8] == 1:
            list2.append(i)
            test_x2.append(x_test_rf[i])
            test_y2.append([0, 0, 1, 0, 0])
            # y2_label.append(3)
            # y2_label.append('mirai_syn')
        elif y_test_rf[i][9] == 1:
            list2.append(i)
            test_x2.append(x_test_rf[i])
            test_y2.append([0, 0, 0, 1, 0])
            # y2_label.append(4)
            # y2_label.append('mirai_udp')
        elif y_test_rf[i][10] == 1:
            list2.append(i)
            test_x2.append(x_test_rf[i])
            test_y2.append([0, 0, 0, 0, 1])
            # y2_label.append(5)
            # y2_label.append('mirai_udpplain')

test_x1 = np.array(test_x1)
test_y1 = np.array(test_y1)
test_x2 = np.array(test_x2)
test_y2 = np.array(test_y2)

with open('./model/G_model.pickle', 'rb') as f:
    clf_gafgyt = pickle.load(f)
with open('./model/M_model.pickle', 'rb') as f:
    clf_mirai = pickle.load(f)

skb_gafgyt = pickle.load(open('./model/G_skb.pkl', 'rb'))
skb_mirai = pickle.load(open('./model/M_skb.pkl', 'rb'))
test_x1 = skb_gafgyt.transform(test_x1)
test_x2 = skb_mirai.transform(test_x2)
scores_gafgyt = skb_gafgyt.scores_
indices_gafgyt = np.argsort(scores_gafgyt)[::-1]
k_best_features_gafgyt = list(data.columns.values[indices_gafgyt[0:12]])
print('k best features are: ', k_best_features_gafgyt)
scores_mirai = skb_mirai.scores_
indices_mirai = np.argsort(scores_mirai)[::-1]
k_best_features_mirai = list(data.columns.values[indices_mirai[0:12]])
print('k best features are: ', k_best_features_mirai)

t_gafgyt = pickle.load(open('./model/G_t.pkl', 'rb'))
t_mirai = pickle.load(open('./model/M_t.pkl', 'rb'))

test_x1 = t_gafgyt.transform(test_x1)
test_x2 = t_mirai.transform(test_x2)

y_pred_gafgyt = clf_gafgyt.predict(test_x1)
accuracy_gafgyt = clf_gafgyt.score(test_x1, test_y1)
print("gafgyt botnet Accuracy: {:.2f}%".format(accuracy_gafgyt * 100))
y_pred_mirai = clf_mirai.predict(test_x2)
accuracy_mirai = clf_mirai.score(test_x2, test_y2)
print("mirai botnet Accuracy: {:.2f}%".format(accuracy_mirai * 100))

y_pred_cm = np.argmax(y_pred_gafgyt, axis=1)
y_test_cm = np.argmax(test_y1, axis=1)
cm = confusion_matrix(y_test_cm, y_pred_cm)
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(5, 5)
label = ['gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp', 'gafgyt_udp']
plt.figure(figsize=(11, 11))
sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, fmt='', cmap="Blues", vmin=0.2, cbar=True,
            annot_kws={'size': 11}, cbar_kws={'pad': 0.02})
plt.title('Confusion Matrix for ' + 'Gafgyt' + ' Model', fontsize='18')
plt.ylabel('True Class', fontsize=16)
plt.xlabel('Predicted Class', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./ans/' + 'gafgyt' + '_CM.png', dpi=600, bbox_inches='tight')
plt.show()

print(classification_report(y_test_cm, y_pred_cm,
                            target_names=['gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp', 'gafgyt_udp']))
with open('./ans/' + 'gafgyt' + '_CR.txt', 'a') as f:
    f.write(classification_report(y_test_cm, y_pred_cm,
                                  target_names=['gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp',
                                                'gafgyt_udp']))

y_pred_cm = np.argmax(y_pred_mirai, axis=1)
y_test_cm = np.argmax(test_y2, axis=1)
cm = confusion_matrix(y_test_cm, y_pred_cm)
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(5, 5)
label = ['mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain']
plt.figure(figsize=(11, 11))
sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, fmt='', cmap="Blues", vmin=0.2, cbar=True,
            annot_kws={'size': 11}, cbar_kws={'pad': 0.02})
plt.title('Confusion Matrix for ' + 'Mirai' + ' Model', fontsize='18')
plt.ylabel('True Class', fontsize=16)
plt.xlabel('Predicted Class', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./ans/' + 'mirai' + '_CM.png', dpi=600, bbox_inches='tight')
plt.show()

print(classification_report(y_test_cm, y_pred_cm,
                            target_names=['mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain']))
with open('./ans/' + 'mirai' + '_CR.txt', 'a') as f:
    f.write(classification_report(y_test_cm, y_pred_cm,
                                  target_names=['mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp', 'mirai_udpplain']))

y_pred_all = []
g = 0
m = 0
for i in range(len(y_pred)):
    if y_pred[i][1] == np.max(y_pred[i], axis=0):
        if y_test_rf[i][1] == 1 or y_test_rf[i][2] == 1 or y_test_rf[i][3] == 1 or y_test_rf[i][4] == 1 or \
                y_test_rf[i][5] == 1:
            if y_pred_gafgyt[g][0] == np.max(y_pred_gafgyt[g], axis=0):
                y_pred_all.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif y_pred_gafgyt[g][1] == np.max(y_pred_gafgyt[g], axis=0):
                y_pred_all.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif y_pred_gafgyt[g][2] == np.max(y_pred_gafgyt[g], axis=0):
                y_pred_all.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif y_pred_gafgyt[g][3] == np.max(y_pred_gafgyt[g], axis=0):
                y_pred_all.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif y_pred_gafgyt[g][4] == np.max(y_pred_gafgyt[g], axis=0):
                y_pred_all.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            g = g + 1
        else:
            y_pred_all.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif y_pred[i][2] == np.max(y_pred[i], axis=0):
        if y_test_rf[i][6] == 1 or y_test_rf[i][7] == 1 or y_test_rf[i][8] == 1 or y_test_rf[i][9] == 1 or \
                y_test_rf[i][10] == 1:
            if y_pred_mirai[m][0] == np.max(y_pred_mirai[m], axis=0):
                y_pred_all.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif y_pred_mirai[m][1] == np.max(y_pred_mirai[m], axis=0):
                y_pred_all.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif y_pred_mirai[m][2] == np.max(y_pred_mirai[m], axis=0):
                y_pred_all.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif y_pred_mirai[m][3] == np.max(y_pred_mirai[m], axis=0):
                y_pred_all.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif y_pred_mirai[m][4] == np.max(y_pred_mirai[m], axis=0):
                y_pred_all.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
            m = m + 1
        else:
            y_pred_all.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        y_pred_all.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
y_pred_all = np.array(y_pred_all)

y_pred_cm = np.argmax(y_pred_all, axis=1)
y_test_cm = np.argmax(y_test_rf, axis=1)
cm = confusion_matrix(y_test_cm, y_pred_cm)
group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
labels = np.asarray(labels).reshape(11, 11)
label = ['benign', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp', 'gafgyt_udp', 'mirai_ack', 'mirai_scan',
         'mirai_syn', 'mirai_udp', 'mirai_udpplain']
plt.figure(figsize=(11, 11))
sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, fmt='', cmap="Blues", vmin=0.2, cbar=True,
            annot_kws={'size': 11}, cbar_kws={'pad': 0.02})
plt.title('Confusion Matrix for ' + 'Proposed Method' + ' Model', fontsize='18')
plt.ylabel('True Class', fontsize=16)
plt.xlabel('Predicted Class', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('./ans/' + 'Proposed Method' + '_CM.png', dpi=600, bbox_inches='tight')
plt.show()

print(classification_report(y_test_cm, y_pred_cm,
                            target_names=['benign', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp',
                                          'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp',
                                          'mirai_udpplain']))
with open('./ans/' + 'Proposed Method' + '_CR.txt', 'a') as f:
    f.write(classification_report(y_test_cm, y_pred_cm,
                                  target_names=['benign', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp',
                                                'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udp',
                                                'mirai_udpplain']))
