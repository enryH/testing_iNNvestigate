import warnings
warnings.simplefilter('ignore')

import imp
import matplotlib.pyplot as plot
import numpy as np
import os

import keras
import keras.backend
import keras.layers
import keras.models
import keras.utils

import innvestigate
import innvestigate.utils as iutils

# Use utility libraries to focus on relevant iNNvestigate routines.
mnistutils = imp.load_source(name="utils_mnist", pathname="utils_mnist.py")

data_not_preprocessed = mnistutils.fetch_data()

# Create preprocessing functions
input_range = [-1, 1]
preprocess, revert_preprocessing = mnistutils.create_preprocessing_f(data_not_preprocessed[0], input_range)

# Preprocess data
data = (
    preprocess(data_not_preprocessed[0]), keras.utils.to_categorical(data_not_preprocessed[1], 10),
    preprocess(data_not_preprocessed[2]), keras.utils.to_categorical(data_not_preprocessed[3], 10),
)

if keras.backend.image_data_format == "channels_first":
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)

# and to now create and train a CNN model:
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same',
                                     input_shape=input_shape),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(data[0], data[1], epochs=3, batch_size=128)

scores = model.evaluate(data[2], data[3], batch_size=128)
print("Scores on test set: loss=%s accuracy=%s" % tuple(scores))

image = data[2][7:8]

plot.imshow(image.squeeze(), cmap='gray', interpolation='nearest')
plot.show()

# Stripping the softmax activation from the model
model_wo_sm = iutils.keras.graph.model_wo_softmax(model)

# Create analyzer
analyzer = innvestigate.create_analyzer("deep_taylor", model_wo_sm)

# Applying the analyzer
analysis = analyzer.analyze(image)

# Check Conservation
scores = model_wo_sm.predict(image)
print("Maximum Score: {:.3f} with label {}".format(scores.max(), scores.argmax()))
print("sum of relevances assigned to inputs: {:.3f}".format(analysis.sum()))
try:
    assert abs(scores.max() - analysis.sum()) < 0.001
except AssertionError:
    print("not equal...")
# Biases are included and conversation property fails

# LRP-Alpha_1-Beta_0 without biases is z+ rule
from innvestigate.analyzer.relevance_based.relevance_analyzer import LRPAlpha1Beta0IgnoreBias
analyzer = LRPAlpha1Beta0IgnoreBias(model_wo_sm)

# Applying the analyzer
analysis = analyzer.analyze(image)

# Check Conservation
scores = model_wo_sm.predict(image)
print("Maximum Score: {:.3f} with label {}".format(scores.max(), scores.argmax()))
print("sum of relevances assigned to inputs: {:.3f}".format(analysis.sum()))
assert abs(scores.max() - analysis.sum()) < 0.001
