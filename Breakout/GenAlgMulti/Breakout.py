import gym
import numpy as np

from GeneticAlgorithemFast import*
from GeneToNetwork import*
import sys
import os
args=sys.argv
# print(args)
if len(args)<3:
    args=["Breakout.py",50,100,1] #iterations then generation size

args[1]=int(args[1])
args[2]=int(args[2])
args[3]=int(args[3])


"""
IDEA: Run several games on each person in a generation
To Do: make it so I can close my computer and have it run

"""



def mutation(new,rate):
    for n, j in enumerate(new):
        if uniform(0, 1) <= rate:
            new[n] = uniform(-1, 1)
    return new


def reward(gene):
    env=gym.make("Breakout-ram-v0",frameskip=1)

    rewards=[]
    for i in gene:
        n=network(i["gene"],[128,30,30,2])
        env.reset()
        env.step(1)
        env.step(1)
        ob, reward, done, info = env.step(1)
        score1=0
        score2=0
        # moves={0:0,1:0,2:0,3:0}
        prevob = ob
        while True:
            # env.render()
            ob=np.array([ob])
            prevlives=info["ale.lives"]
            move=(n.predict(ob,0,prevob))+2
            move=np.argmax(move)
            # moves[move]+=1
            prevob=ob


            ob, reward, done,info=env.step(move)
            if reward!=0:
                # if reward>1:
                score2-=reward

            if info['ale.lives']<prevlives:
                # score1-=1
                break
            if done==True:
                break
        env.close()
        rewards.append({"score":score1+score2,"index":i["index"]})

    return rewards


if __name__=="__main__":
    g=Algorithem(4800,args[2],reward,mutation)
    dropoutrate=0
    genes, best, topscore, lowscore = g.generation(poolsize=args[3])

    mut=0.008
    for i in range(args[1]):
        genes, best, topscore,lowscore =g.generation(genes,best,mut,args[3])
        # if topscore<-2:
        #     if mut>0.0005:
        #         mut-=0.0005

        if i%1==0:
            print ("Iteration: "+str(i), "Top Score: "+str(topscore), "Low Score: "+str(lowscore), "Mutation Rate: "+str(mut))
            if dropoutrate<0.5:
                dropoutrate+=0.1
    np.save("save.npy",np.array(genes[best]))
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

