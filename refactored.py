import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
# tfds.disable_progress_bar()

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)

encoder = info.features['text'].encoder
encoder.subwords[:20]

train_batches = train_data.shuffle(1000).padded_batch(10)
test_batches = test_data.shuffle(1000).padded_batch(10)

train_batch, train_labels = next(iter(train_batches))
train_batch.numpy()

vocab_size = 1000
dimentions = 5
embedding_dim=16

embedding_layer = layers.Embedding(vocab_size, dimentions)


model = tf.keras.models.Sequential()
model.add(layers.Embedding(encoder.vocab_size, embedding_dim))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1))

model.summary()

epochs = 10
learningRate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate, name='Adam')

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=epochs,
    validation_data=test_batches, validation_steps=20)
