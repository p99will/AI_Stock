import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from matplotlib import pyplot as plt
import random

PF = "StockData - "

STOCK_FILES = ['apple.csv','microsoft.csv','samsung.csv']
memory = 50
MEASURING_AGAINST = 'apple.csv'


def generageData():
    data = []
    labels = pd.read_csv(PF+MEASURING_AGAINST)[['AVG']].values.tolist()
    for i in STOCK_FILES:
        df=pd.read_csv(PF+i).values.tolist()
        close = []
        avg = []
        for x in df:
            close.append(x[3])
            # avg.append(x[3])
        # close = tf.keras.utils.normalize(close)[0]
        data.append(close)

    lowestCnount = 999
    for i in data:
        if(len(i) < lowestCnount):
            lowestCnount = len(i)
    # print(lowestCnount)


    startNum = 0
    samples = []
    for i in range(lowestCnount-memory):
        unit = []
        for y in range(len(data)):
            unit.append([])
            for x in range(i,memory+i):
                unit[y].append(data[y][x])
        # sample_label.append(labels[i+memory])
        samples.append([unit,labels[i+memory-1]])
    # random.shuffle(samples)
    features = []
    labels = []
    for i in samples:
        _x= []
        for x in i[0]:
            _x.append(x)
        #     for y in x:
        #         _x.append(y)
        features.append(_x)
        lab = int(i[1][0]*10)
        if(lab < 0):
            lab = 0
        else:
            lab = 1
        labels.append(lab)

    return features, labels

def create_model(my_learning_rate,outputs):
    """Create and compile a deep neural net."""

    # All models in this course are sequential.
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=memory,
                                  input_shape=(3,memory)))
    model.add(tf.keras.layers.Dense(units=500, activation='relu'))
    model.add(tf.keras.layers.Dense(units=500, activation='sigmoid' ))
    model.add(tf.keras.layers.Dense(units=500, activation='sigmoid' ))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
            kernel_regularizer=regularizers.l1_l2(l1=0.1, l2=0.5),
        ))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                loss="binary_crossentropy",
                metrics=['accuracy'])

    return model


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                      epochs=epochs, shuffle=True,
                      validation_split=validation_split)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

learning_rate = 0.001
epochs = 8000
batch_size = 100
validation_split = 0.2

# Establish the model's topography.

# Train the model on the normalized training set.
features, labels = generageData()
labels = np.array(labels)
features = tf.constant(features)
# features = tf.math.l2_normalize(features)

outputs = len(set(labels))

my_model = create_model(learning_rate,outputs)



epochs, hist = train_model(my_model, features, labels,
                           epochs, batch_size, validation_split)
