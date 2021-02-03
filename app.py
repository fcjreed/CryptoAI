import json
import matplotlib.pyplot as plt
import normalizer as norm
import pandas as pd
import tensorflow as tf
from backfiller import BackFiller
from tensorflow import keras
from tensorflow.keras import layers

backfill = BackFiller()

train_data, test_data = backfill.load_training_data()

MULTIPLIER = round(len(train_data)/10)
BATCH_SIZE = MULTIPLIER if len(train_data) <= 320 else 32
EPOCHS = 100

input_data, output = norm.setup_transposed_data(train_data)
test_input_data, test_output_data = norm.setup_transposed_data(test_data)
counter = 0
while counter < len(input_data):
    input_data[counter] = norm.normalize_zscore(input_data[counter])
    counter += 1
counter = 0
while counter < len(test_input_data):
    test_input_data[counter] = norm.normalize_zscore(test_input_data[counter])
    counter += 1
#output = norm.normalize_zscore(output)

with open("datasets/normalized_train_data.txt", "w") as filehandle:
    json.dump(input_data, filehandle)

with open("datasets/normalized_test_data.txt", "w") as filehandle:
    json.dump(test_input_data, filehandle)

# model = keras.Sequential([
#     layers.Dense(4, input_shape=[4,1], activation=tf.nn.relu),
#     layers.Dense(1, activation=tf.nn.softmax)
# ])

# model.compile(optimizer='adam',
#             loss=keras.losses.SparseCategoricalCrossentropy(),
#             metrics=['accuracy'])

# history = model.fit(input_data, output, epochs=EPOCHS, verbose=False)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(history.history['loss'])