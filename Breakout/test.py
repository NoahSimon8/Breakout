import gym
import random
import numpy as np
from time import*
from Faster.GeneticAlgorithemFast import*
from Faster.GeneToNetwork import*
import os


#
# n=network([1 for i in range(8)],[2,2,2])
# env=gym.make("Breakout-ram-v0",frameskip=1)
# env.reset()
# while True:
#     env.render()
#     env.step(int(input("move")))
n=network(layers=[128,10,20,2])
print(n.numGenes())