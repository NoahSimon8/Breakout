from time import*
import gym
import random
import numpy as np
from GeneticAlgorithemFast import*
from GeneToNetwork import*
env=gym.make("Breakout-ram-v0")

weights=np.load("save.npy")
print(weights)
# print(weights.shape)

n=network(weights,[128,10,10,4])

for i in range(20):
    ob = env.reset()
    env.step(1)
    env.step(1)
    ob, reward, done, info = env.step(1)

    while True:
        sleep(0.02)
        ob=np.array([ob])
        # print(ob.shape)
        env.render()
        ob, reward, done,info=env.step(np.argmax(n.predict(ob,0,ob)))
        if info['ale.lives']==4:
            env.close()
            break

# def reward(gene):
#     rewards=[]
#     # print(len(gene))
#     for i in gene:
#         n=network(i,[128,30,30,2])
#         env.reset()
#         env.step(1)
#         env.step(1)
#         ob, reward, done, info = env.step(1)
#         score1=0
#         score2=0
#         moves={0:0,1:0,2:0,3:0}
#         prevob = ob
#         while True:
#             # env.render()
#             ob=np.array([ob])
#             prevlives=info["ale.lives"]
#             move=np.argmax(n.predict(ob,0,prevob))+2
#             moves[move]+=1
#             prevob=ob
#
#
#             ob, reward, done,info=env.step(move)
#             if reward!=0:
#                 # print("YES",reward)
#                 # if reward>1:
#                 #     print(moves)
#                 score2-=reward
#                 # print(moves)
#
#             if info['ale.lives']<prevlives:
#                 # score1-=1
#                 # print("lost")
#                 break
#             if done==True:
#                 # print("end")
#                 break
#         env.close()
#         rewards.append(score1+score2)
#         # print(score1+score2)
#     return rewards
#
# print(reward(weights))