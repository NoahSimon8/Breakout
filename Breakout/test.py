# import gym
# import random
# import numpy as np
# from time import*
# # from Faster.GeneticAlgorithemFast import*
# # from Faster.GeneToNetwork import*
# import os
#
# env = gym.make("Breakout-ram-v0", frameskip=1)
# one=np.array(env.reset()).astype("int8")
# ob, reward, done, info = env.step(1)
# two=np.array(ob).astype("int8")
# # print(one-two)
# ob, reward, done, info = env.step(2)
# three=np.array(ob).astype("int8")
# ob, reward, done, info = env.step(2)
# four=np.array(ob).astype("int8")
# # print(two-three)
# # print(one)
# # print(two)
# # print(three.dtype)
# print(two-three)
# print(three-four)



import gym
import random
import numpy as np

env = gym.make("Breakout-ram-v0", frameskip=1)
one=np.array(env.reset()).astype("int8")
ob, reward, done, info = env.step(1)
ob=np.array([ob])
print(ob/255)