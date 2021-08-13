import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# model=keras.Sequential([
#     keras.layers.Conv2D(10,(4,1),activation="sigmoid",input_shape=(128,1,1,)),
#     keras.layers.MaxPooling2D((2,1)),
#     keras.layers.Conv2D(10,(4,1),activation="sigmoid"),
#     keras.layers.MaxPooling2D((2,1)),
#     keras.layers.Flatten(),
#
#     keras.layers.Dense(10,activation="sigmoid"),
#     keras.layers.Dense(10,activation="softmax")])
#
# model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

def numweights(shape):
    total=1
    for i in range(len(shape)):
        total*=shape[i]
    return total


def networktogene(model): #network to gene
    megalist=[]
    for i in model.layers:
        w=i.get_weights()
        if w!=[]:
            shape=w[0].shape

            wsize=numweights(shape) #how many weights are in the total layer

            megalist+=(w[0].reshape(1,wsize).tolist()[0]) #combines them
    return megalist


def genetonetwork(model,lst):
    layerindex=0
    weightcount = 0  #how far into the mega list we are
    for i in range(len(model.layers)):
        w=model.get_layer(index=layerindex).get_weights()

        if w!=[]:
            shape=w[0].shape #the shape the weighst will need to be in




            wsize=numweights(shape) #how many weights are in the total layer

            weights=lst[weightcount:weightcount+wsize] #splices into the correct layer
            weights=np.array(weights)
            weights=weights.reshape(shape) #sets into the correct layer shape
            WandB=[weights,w[1]] #weights and biasies


            weightcount+=wsize

            model.get_layer(index=layerindex).set_weights(WandB)
            layerindex+=1
    return model

