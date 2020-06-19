import tensorflow as tf

class ezTF:
    model = None
    hastrained = False
    nodes=0
    layers=0
    epoc = 0
    learningRate=0

    def __init__(self,inputs,outputs,nodes,layers,epoc):
        self.epoc = epoc
        self.nodes = nodes
        self.layers = layers
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(inputs, activation='relu'))
        for i in (0,layers):
            self.model.add(tf.keras.layers.Dense(nodes, activation='relu'))
        self.model.add(tf.keras.layers.Dense(outputs,activation=tf.nn.softmax))

    def compile(self):
        self.model.compile(optimizer='adam',
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])

    def train(self,training_data,training_labels):
        self.hastrained = True
        return self.model.fit(training_data,training_labels,epochs=self.epoc)

    def evaluate(self,test_data,test_labels):
        return self.model.evaluate(test_data,test_labels)

    def predict(self,test_data):
        return self.model.predict(test_data)
