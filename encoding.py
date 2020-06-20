import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import os

DEBUG = True
VERBOS = 0 # 0 = All, 1 = Info , 2 = Important only, 3 = Errors only.

def log(output,verbous_level):
    if(DEBUG):
        if(VERBOS <= verbous_level):
            print(output)

class WordEncoder():
    all_labeled_data = None
    vocab_size = None
    vocabulary_set = None
    encoder = None

    def __init__(self,all_labeled_data):
        log("Initing Tokenizer", 0)
        tokenizer = tfds.features.text.Tokenizer()
        vocabulary_set = set()
        log("Tokenizing all labeled data",1)
        for text_tensor, _ in all_labeled_data:
            some_tokens = tokenizer.tokenize(text_tensor.numpy())
            vocabulary_set.update(some_tokens)
        vocab_size = len(vocabulary_set)
        log("Vocab size: " + str(vocab_size),1)
        self.encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)
        self.vocab_size = vocab_size
        self.vocabulary_set = vocabulary_set

    def testEncode(self,text=''):
        if(text=''):
            log("No example text provided, Fetching one from data...",2)
            text = next(iter(self.all_labeled_data))[0].numpy()
        print("Given text: ")
        print(text)
        print("\nEncoded Text:")
        print(self.encoder.encode(text))

    def __labeler(self,example, index):
        return example, tf.cast(index, tf.int64)

    def encode(self,text_tensor, label):
        encoded_text = self.encoder.encode(text_tensor.numpy())
        return encoded_text, label

    def readText(self,FILE_NAMES):
        successful = False
        labeled_data_sets = []
        try:
            log("Reading files:",1)
            for i, file_name in enumerate(FILE_NAMES):
                log("Reading file: " + file_name, 0)
                lines_dataset = tf.data.TextLineDataset(os.path.join(file_name))
                log("Labeling file: " + file_name, 0)
                labeled_dataset = lines_dataset.map(lambda ex: self.__labeler(ex, i))
                labeled_data_sets.append(labeled_dataset)
        except Exception as e:
            log("Error while reading files: ",3)
            log(e,3)
        else:
            successful = True
            BUFFER_SIZE = 50000
            BATCH_SIZE = 64
            TAKE_SIZE = 5000

            log("Combinding all files into one",1)
            all_labeled_data = labeled_data_sets[0]
            for labeled_dataset in labeled_data_sets[1:]:
                all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

            log("Shuffeling dataset",1)
            self.all_labeled_data = all_labeled_data.shuffle(
                BUFFER_SIZE, reshuffle_each_iteration=False)
        return successful




class WordEmbeddingModel():
    model = None

    # Note: 'nodes' will be a 2d list, each item will be a new layer
    #       having the item's value as the number of nodes.
    #       EG: [4,3] - Layer 1: 4 nodes, Layer2: 3 nodes.
    def __init__(self,vocab_size=1000,dimentions=5,nodes=[16]):
        model = tf.keras.models.Sequential()
        model.add(layers.Embedding(vocab_size, dimentions))
        model.add(layers.GlobalAveragePooling1D())
        for i in nodes:
            model.add(layers.Dense(i, activation='relu'))
        model.add(layers.Dense(1))
        self.model = model

    def compile(self,learningRate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate,
            name='Adam')

        self.model.compile(optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])

        return self.model

    def train(self,)
