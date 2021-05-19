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

n=network(weights,[128,30,30,2])
ob = env.reset()
while True:
    sleep(0.02)
    ob=np.array([ob])
    env.render()
    ob, reward, done,info=env.step(np.argmax(n.predict(ob,0,ob)))
    if info['ale.lives']==4:
        break
