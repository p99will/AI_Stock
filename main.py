import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt

class MyModel:
    model = tf.keras.models.Sequential()
    hastrained = False
    nodes=0
    layers=0
    epoc = 0
    learningRate=0

    def __init__(self,inputs,outputs,nodes,layers):
        self.nodes = nodes
        self.layers = layers
        self.model.add(tf.keras.layers.Dense(inputs, activation='relu'))
        for i in (0,layers):
            self.model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        self.model.add(tf.keras.layers.Dense(outputs,activation=tf.nn.softmax))

    def compile(self,my_metrics,learningRate):
        self.learningRate = learningRate
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learningRate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=my_metrics)

    def train(self, model, dataset, epochs, label_name)
                batch_size=None, shuffle=True):
        features = training_data
        label = training_labels
        history = model.fit(x=features, y=label, batch_size=batch_size,
                      epochs=epochs, shuffle=shuffle)
        self.hastrained = True
        return self.model.fit(training_data,training_labels,epochs=self.epoc)

    def evaluate(self,test_data,test_labels):
        return self.model.evaluate(test_data,test_labels)

    def predict(self,test_data):
        return self.model.predict(test_data)
