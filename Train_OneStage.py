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
    MaxPooling1D, Attention, GRU, GRUCell, Bidirectional
import tensorflow as tf

import pickle
import seaborn as sns
from itertools import cycle
import scipy.io as scio

modelName = '1DCNN+BiGRU'
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
g_c['type'] = 'gafgyt'
g_j['type'] = 'gafgyt'
g_s['type'] = 'gafgyt'
g_t['type'] = 'gafgyt'
g_u['type'] = 'gafgyt'
m_a['type'] = 'mirai'
m_sc['type'] = 'mirai'
m_sy['type'] = 'mirai'
m_u['type'] = 'mirai'
m_u_p['type'] = 'mirai'

data = pd.concat([benign, g_c, g_j, g_s, g_t, g_u, m_a, m_sc, m_sy, m_u, m_u_p], axis=0, sort=False, ignore_index=True)

labels_full = pd.get_dummies(data['type'], prefix='type')
labels_full.head()
labels = labels_full.values

data = data.drop(columns='type')
data_st = data.copy()
data_st = data_st.values

train_data_st = data_st
x_train, x_test, y_train, y_test = train_test_split(train_data_st, labels, test_size=0.2)

t = StandardScaler()
x_train = t.fit_transform(x_train)
x_test = t.transform(x_test)
pickle.dump(t, open('./model/' + modelName + '_t.pkl', 'wb'))

x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.125)
x_train_cnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_validate_cnn = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1], 1))

inp = Input(shape=(train_data_st.shape[1], 1))
L1 = Conv1D(filters=64, kernel_size=5, strides=1, padding='same')(inp)
L2 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(L1)
# L3 = Attention()([L2, L2, L2])
L3 = Bidirectional(GRU(32, return_sequences=True))(L2)
L4 = Bidirectional(GRU(16, return_sequences=True))(L3)
L5 = Flatten()(L4)
L6 = Dense(128, activation='relu')(L5)
L7 = Dense(labels.shape[1], activation='softmax')(L6)
model = Model(inputs=inp, outputs=L7)
modelName = '1DCNN+BiGRU'
model.summary()

adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, lr=0.00001)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)
checkpoint = ModelCheckpoint('./model/' + modelName + '.h5', monitor='val_loss', mode='min', save_best_only=True,
                             # save_weights_only=True,
                             verbose=1)

epochs = 30
batch_size = 512
history = model.fit(x_train_cnn, y_train, batch_size=batch_size, steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs, validation_data=(x_validate_cnn, y_validate),
                    callbacks=[learning_rate_reduction, checkpoint])


def plot_model_history(model_history):
    plt.plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'], '--*',
             color='red')
    plt.plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'], '--^',
             color='purple')
    plt.title('Model ' + modelName + ' Accuracy', fontsize='18')
    plt.xlabel('Epoch', fontsize='16')
    plt.ylabel('Accuracy', fontsize='16')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid('on')
    plt.savefig('./ans/' + modelName + ' Accuracy' + '.jpg', dpi=600, bbox_inches='tight')
    plt.show()

    plt.plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'], '--x',
             color='blue')
    plt.plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'], '--D',
             color='green')
    plt.title('Model ' + modelName + ' Loss', fontsize='18')
    plt.xlabel('Epoch', fontsize='16')
    plt.ylabel('Loss', fontsize='16')
    plt.legend(['Train', 'Validation'], loc='best')
    plt.grid('on')
    plt.savefig('./ans/' + modelName + ' Loss' + '.jpg', dpi=600, bbox_inches='tight')
    plt.show()


plot_model_history(history)
with open('./ans/History_' + modelName, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
