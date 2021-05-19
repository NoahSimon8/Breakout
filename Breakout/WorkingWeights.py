from time import*
import gym
import random
import numpy as np
from GenAlg.GeneticAlgorithemFast import*
from GenAlg.GeneToNetwork import*
env=gym.make("Breakout-ram-v0")

weights=np.load("save.npy")
print(weights)
print(weights.shape)