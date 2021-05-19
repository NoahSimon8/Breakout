import gym
from GenAlg.GeneticAlgorithemFast import*
from GenAlg.GeneToNetwork import*
import sys
import os
args=sys.argv

if len(args)<2:
    args=[300,300] #iterations then generation size
args[1]=int(args[1])
args[2]=int(args[2])


def mutation(new,rate):
    for n, j in enumerate(new):
        if uniform(0, 1) <= rate:
            new[n] = uniform(-1, 1)
    return new


def reward(gene):
    rewards=[]
    # print(len(gene))
    for i in gene:
        n=network(i,[128,30,30,2])
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
                # print("end")
                break
        env.close()
        rewards.append(score1+score2)
        # print(score1+score2)
    return rewards

env=gym.make("Breakout-ram-v0",frameskip=1)
g=Algorithem(4800,args[2],reward,mutation)
dropoutrate=0
print("LOOP")
# try:
#     genes=np.load("save.npy")
#     best=0
#     print("resuming")
genes, best, topscore, lowscore = g.generation()
mut=0.02
for i in range(args[1]):
    print(args[1],args[2])
    genes, best, topscore,lowscore =g.generation(genes,best,mut)
    if topscore<-2:
        mut=0.0005

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

env.close()