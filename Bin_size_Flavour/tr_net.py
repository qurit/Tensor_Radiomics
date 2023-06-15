
import os
import numpy as np
import tensorflow as tf
import keras

from keras.utils import normalize
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, MaxPooling2D, concatenate, BatchNormalization, Dropout, Lambda, Dense
from keras.models import load_model
from keras.metrics import MeanIoU
from keras import backend as K

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import imblearn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE

import matplotlib.pyplot as plt

from datetime import datetime 

print(tf.__version__)


from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

root_dir = '/home/ndubljevic/TR-NET'

y = np.load(os.path.join(root_dir, 'class_labels.npy'))

# Don't be confused, 'feat' really represents flavours. "feat_01_X" is a
# numpy array of all the features values for flavour 1

feat_01_X = np.load(os.path.join(root_dir, 'feat_01_X.npy'))
feat_size_01 = feat_01_X.shape[-1]
feat_02_X = np.load(os.path.join(root_dir, 'feat_02_X.npy'))
feat_size_02 = feat_02_X.shape[-1]
feat_03_X = np.load(os.path.join(root_dir, 'feat_03_X.npy'))
feat_size_03 = feat_03_X.shape[-1]
feat_04_X = np.load(os.path.join(root_dir, 'feat_04_X.npy'))
feat_size_04 = feat_04_X.shape[-1]
feat_05_X = np.load(os.path.join(root_dir, 'feat_05_X.npy'))
feat_size_05 = feat_05_X.shape[-1]
feat_06_X = np.load(os.path.join(root_dir, 'feat_06_X.npy'))
feat_size_06 = feat_06_X.shape[-1]
feat_07_X = np.load(os.path.join(root_dir, 'feat_07_X.npy'))
feat_size_07 = feat_07_X.shape[-1]
feat_08_X = np.load(os.path.join(root_dir, 'feat_08_X.npy'))
feat_size_08 = feat_08_X.shape[-1]
feat_09_X = np.load(os.path.join(root_dir, 'feat_09_X.npy'))
feat_size_09 = feat_09_X.shape[-1]
feat_10_X = np.load(os.path.join(root_dir, 'feat_10_X.npy'))
feat_size_10 = feat_10_X.shape[-1]

# Manage Train / Test / Validation sets

X = np.concatenate([feat_01_X, feat_02_X, feat_03_X, feat_04_X, feat_05_X, feat_06_X, feat_07_X, feat_08_X ,feat_09_X, feat_10_X], axis=1)
y = np.load(os.path.join(root_dir, 'class_labels.npy'))

X_train_, X_test, y_train_, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=11)
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, stratify=y_train_, test_size=0.3,random_state=11)

oversample = SVMSMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

feature_size = feat_01_X.shape[1]

# ================================================================

feat_01_X_train = X_train[:, 0 : feat_size_01]
feat_01_X_val = X_val[:, 0:feature_size]
feat_01_X_test = X_test[:, 0 : feat_size_01]
start_point = feat_size_01

feat_02_X_train = X_train[:, start_point : start_point + feat_size_02]
feat_02_X_val = X_val[:, start_point : start_point + feat_size_02]
feat_02_X_test = X_test[:, start_point : start_point + feat_size_02]
start_point += feat_size_02

feat_03_X_train = X_train[:, start_point : start_point + feat_size_03]
feat_03_X_val = X_val[:, start_point : start_point + feat_size_03]
feat_03_X_test = X_test[:, start_point : start_point + feat_size_03]
start_point += feat_size_03

feat_04_X_train = X_train[:, start_point : start_point + feat_size_04]
feat_04_X_val = X_val[:, start_point : start_point + feat_size_04]
feat_04_X_test = X_test[:, start_point : start_point + feat_size_04]
start_point += feat_size_04

feat_05_X_train = X_train[:, start_point : start_point + feat_size_05]
feat_05_X_val = X_val[:, start_point : start_point + feat_size_05]
feat_05_X_test = X_test[:, start_point : start_point + feat_size_05]
start_point += feat_size_05

feat_06_X_train = X_train[:, start_point : start_point + feat_size_06]
feat_06_X_val = X_val[:, start_point : start_point + feat_size_06]
feat_06_X_test = X_test[:, start_point : start_point + feat_size_06]
start_point += feat_size_06

feat_07_X_train = X_train[:, start_point : start_point + feat_size_07]
feat_07_X_val = X_val[:, start_point : start_point + feat_size_07]
feat_07_X_test = X_test[:, start_point : start_point + feat_size_07]
start_point += feat_size_07

feat_08_X_train = X_train[:, start_point : start_point + feat_size_08]
feat_08_X_val = X_val[:, start_point : start_point + feat_size_08]
feat_08_X_test = X_test[:, start_point : start_point + feat_size_08]
start_point += feat_size_08

feat_09_X_train = X_train[:, start_point : start_point + feat_size_09]
feat_09_X_val = X_val[:, start_point : start_point + feat_size_09]
feat_09_X_test = X_test[:, start_point : start_point + feat_size_09]
start_point += feat_size_09

feat_10_X_train = X_train[:, start_point : start_point + feat_size_10]
feat_10_X_val = X_val[:, start_point : start_point + feat_size_10]
feat_10_X_test = X_test[:, start_point : start_point + feat_size_10]


train_list = [
            feat_01_X_train, 
            feat_02_X_train,
            feat_03_X_train,
            feat_04_X_train,
            feat_05_X_train,
            feat_06_X_train,
            feat_07_X_train,
            feat_08_X_train,
            feat_09_X_train,
            feat_10_X_train,
            ]

val_list = [
            feat_01_X_val, 
            feat_02_X_val,
            feat_03_X_val,
            feat_04_X_val,
            feat_05_X_val,
            feat_06_X_val,
            feat_07_X_val,
            feat_08_X_val,
            feat_09_X_val,
            feat_10_X_val,
            ]

test_list = [
            feat_01_X_test, 
            feat_02_X_test,
            feat_03_X_test,
            feat_04_X_test,
            feat_05_X_test,
            feat_06_X_test,
            feat_07_X_test,
            feat_08_X_test,
            feat_09_X_test,
            feat_10_X_test,
            ]



def TR_Net(input_list, 
            num_features, 
            num_flavors, 
            activation="sigmoid", 
            n_leg_dense_layers=2, 
            n_body_dense_layers=2, 
            leg_layer_size=10, 
            body_layer_size=100):

    """Builds a base RT Net architecture.
    Args:
        
    Returns:
        The keras `Model`.
    """

    K.clear_session()

    legs = list()

    inputs = list()

    for flavor in input_list:
        input = Input(shape=(flavor.shape[-1], ))
        inputs.append(input)
        leg = input
        for _ in range(n_leg_dense_layers):
            leg = Dense(leg_layer_size, activation='selu')(leg)
            # Add BatchNorm ?
            # leg = BatchNormalization()(leg)
            # Add DropOut ?
            leg = Dropout(0.3)(leg)
            # Use 1D_Conv ?
            # Use LSTM ?
        legs.append(leg)

    body = concatenate([leg for leg in legs])

    for _ in range(n_body_dense_layers):
        body = Dense(body_layer_size, activation='selu')(body)
        # Add BatchNorm ?
        # body = BatchNormalization()(body)
        # Add DropOut ?
        body = Dropout(0.3)(body)

    output = Dense(1, activation=activation)(body)
     
    model = Model(inputs=[input for input in inputs], outputs=output)

    return model


print("Train input shape: ", train_list[-1].shape)
print("Val tensor shape: ", val_list[-1].shape)
print("Test tensor shape: ", test_list[-1].shape)

num_features = feature_size
num_flavors = len(train_list)

K.clear_session()

model = TR_Net(train_list, 
                num_features, 
                num_flavors, 
                activation="sigmoid", 
                n_leg_dense_layers=2, 
                n_body_dense_layers=3, 
                leg_layer_size=10, 
                body_layer_size=100)

LR = 0.01
optimizer = keras.optimizers.SGD(learning_rate=LR, momentum=0.9)
total_loss = keras.losses.MeanSquaredError()
METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),
]

model.compile(loss=[total_loss], optimizer=optimizer, metrics=METRICS)
model.summary()
keras.utils.plot_model(model, show_shapes=True, expand_nested=True)


callbacks = [
            keras.callbacks.ModelCheckpoint('./best_model_TR_NET.h5', verbose=1, monitor='val_prc', save_best_only=True, mode='max')
            ]

start_time = datetime.now()

num_epochs = 300

history = model.fit(train_list, 
                    y_train,
                    batch_size=40, 
                    epochs=num_epochs,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=(val_list, y_val))

end_time = datetime.now()

execution_time_unet = end_time - start_time
print("training TR net took: ", execution_time_unet)

model.save('./TR-net_300_epochs.hdf5')

def plot_metrics(history):
  metrics = ['loss', 'prc', 'accuracy', 'precision', 'recall', 'tp', 'fn']
  colors = ['green', 'orange']
  plt.figure(figsize=(20, 15))
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(3,3,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[1], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])

    plt.legend();

  plt.savefig('metrics.png')

plot_metrics(history)

model.load_weights('best_model_TR_NET.h5')
preds = model.predict(test_list)
np.save("tr_preds.npy",preds)

baseline_results = model.evaluate(test_list, y_test,
                                  batch_size=67, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()


TP = baseline_results[1]
FP = baseline_results[2]
TN = baseline_results[3]
FN = baseline_results[4]

print("TP: ", TP)
print("FP: ", FP)
print("TN: ", TN)
print("FN: ", FN)

Balanced_Accuracy = (TP/(TP+FN)+TN/(TN+FP)) / 2

print('Balanced Accuracy of TR-Net = {:0.3f}'.format(Balanced_Accuracy))

recall = TP/(TP+FN)
precision = TP/(TP+FP)
F1 = 2 * (precision * recall) / (precision + recall)

print('F1 score of = {:0.3f}'.format(F1))

def show_metrics_from_ratios(tp, fp, tn, fn):
   
    # True positive rate (sensitivity or recall)
    tpr = tp / (tp + fn)
    # False positive rate (fall-out)
    fpr = fp / (fp + tn)
    # Precision
    precision = tp / (tp + fp)
    # True negatvie tate (specificity)
    tnr = 1 - fpr
    # F1 score
    f1 = 2*tp / (2*tp + fp + fn)
    # ROC-AUC for binary classification
    auc = (tpr+tnr) / 2
    # MCC
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    balacc = (tpr + tnr)/2

    # print("True positive: ", tp)
    # print("False positive: ", fp)
    # print("True negative: ", tn)
    # print("False negative: ", fn)
    print(f"Balacc: {balacc}")
    # print("True positive rate (recall): ", tpr)
    # print("False positive rate: ", fpr)
    # print("Precision: ", precision)
    # print("True negative rate: ", tnr)
    print("F1: ", f1)
    print("ROC-AUC: ", auc)
    # print("MCC: ", mcc)

#show_metrics_from_ratios(TP, FP, TN, FN)

