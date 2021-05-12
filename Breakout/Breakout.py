import gym
import random
import numpy as np
from time import*
from Faster.GeneticAlgorithemFast import*
from Faster.GeneToNetwork import*
import os

"""
Notes:
Ball spawns most but not all of the time
Edit gene size to make it only move left or right


"""

def mutation(new,rate):
    for n, j in enumerate(new):
        if uniform(0, 1) <= rate:
            new[n] = uniform(-1, 1)
    return new


def reward(gene):
    rewards=[]
    # print(len(gene))
    for i in gene:
        n=network(i,[128,10,20,2])
        env.reset()
        env.step(1)
        env.step(1)
        ob, reward, done, info = env.step(1)
        score1=0
        score2=0
        moves={0:0,1:0,2:0,3:0}
        prevob = ob
        while True:
            # env.render()
            ob=np.array([ob])
            prevlives=info["ale.lives"]
            move=np.argmax(n.predict(ob,0,prevob))+2
            moves[move]+=1
            prevob=ob


            ob, reward, done,info=env.step(move)
            if reward!=0:
                # print("YES",reward)
                # if reward>1:
                #     print(moves)
                score2-=reward
                # print(moves)

            if info['ale.lives']<prevlives:
                # score1-=1
                # print("lost")
                break
            if done==True:
                print("end")
                break
        env.close()
        rewards.append(score1+score2)
        # print(score1+score2)
    return rewards

env=gym.make("Breakout-ram-v0",frameskip=1)
g=Algorithem(1520,30,reward,mutation)
dropoutrate=0
print("LOOP")
try:
    genes=np.load("save.npy")
    best=0
    print("resuming")
except:
    print("starting new")
    genes, best, topscore = g.generation()
mut=0.15
for i in range(600):
    print("ITERATION", i)
    genes, best, topscore=g.generation(genes,best,mut)
    if topscore<-2:
        mut=0.0005

    if i%5==0:
        print(i, "Top Score: "+str(topscore))

        if dropoutrate<0.5:
            dropoutrate+=0.1
        # try:
        #     os.remove("save.npy")
        # except:
        #     pass
        # np.save("save.npy",np.array(genes))
print("Top Score: "+str(topscore), "Composition: ", genes[best])





# n=network(genes[best],[128,10,20,2])
# ob = env.reset()
# while True:
#     ob=np.array([ob])
#     env.render()
#     ob, reward, done,info=env.step(np.argmax(n.predict(ob,0)))
#     if info['ale.lives']==4:
#         break


# print(ob.size,reward, info)

env.close()