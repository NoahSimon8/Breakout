import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


model=keras.Sequential([
    keras.layers.Conv2D(10,(1,4),activation="sigmoid",input_shape=(1,128,1,)),
    keras.layers.MaxPooling2D((1,2)),
    keras.layers.Conv2D(10,(1,4),activation="sigmoid"),
    keras.layers.MaxPooling2D((1,2)),
    # keras.layers.Flatten(),

    keras.layers.Dense(10,activation="sigmoid"),
    keras.layers.Dense(10,activation="softmax")])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
# print(model.summary())
print("weights")
print(model.get_layer(name=None,index=1).get_weights())