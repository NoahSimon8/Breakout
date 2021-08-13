import gym
import numpy as np
from GeneticAlgorithemFast import*
from Keras import*
import sys
import os
from tensorflow import keras


# model=keras.Sequential([
#     keras.layers.Conv2D(10,(4,1),activation="sigmoid",input_shape=(128,1,1,)),
#     keras.layers.MaxPooling2D((2,1)),
#     keras.layers.Conv2D(10,(4,1),activation="sigmoid"),
#     keras.layers.MaxPooling2D((2,1)),
#     keras.layers.Flatten(),
#
#     keras.layers.Dense(10,activation="sigmoid"),
#     keras.layers.Dense(4,activation="softmax")])
#
# model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])



model=keras.Sequential([

    keras.layers.Dense(256, activation="ReLU",input_shape=(1,128,)),
    keras.layers.Dense(256, activation="ReLU"),
    keras.layers.Dense(256, activation="ReLU"),
    keras.layers.Dense(256,activation="ReLU"),
    keras.layers.Dense(4,activation="softmax")])

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])




args=sys.argv
# print(args)
if len(args)<3:
    args=["Breakout.py",50,10,1] #iterations then generation size, then #cpus

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
        score = 0
        for n in range(1):
            n=genetonetwork(model,i["gene"])
            env.reset()
            env.step(1)
            env.step(1)
            ob, reward, done, info = env.step(1)

            prevob = ob
            print("WHILE LOOP")
            while True:
                ob=np.array([ob])
                ob=ob.astype("float16")/255
                prevlives=info["ale.lives"]
                ob=ob/255  #standardize


                move=np.argmax(n.predict(ob))

                # move=np.argmax(move)
                # moves[move]+=1

                predicted=(n.predict(ob))

                ob, reward, done,info=env.step(move)

                if reward!=0:
                    score-=reward

                if info['ale.lives']<prevlives:
                    break
                if done==True:
                    break


        env.close()
        rewards.append({"score":score,"index":i["index"]})
    print("DONE")
    return rewards


if __name__=="__main__":
    g=Algorithem(230400,args[2],reward,mutation) #pass in the total number of weights, how many cpus to use, reward function, mutation function
    dropoutrate=0
    genes, best, topscore, lowscore, avgscore = g.generation(poolsize=args[3])

    mut=0.01
    for i in range(args[1]):
        genes, best, topscore,lowscore, avgscore =g.generation(genes,best,mut,args[3])
        # if topscore<-2:
        #     if mut>0.0005:
        #         mut-=0.0005

        if i%1==0:
            print ("Iteration: "+str(i), "Top Score: "+str(topscore),"Average Score: "+str(avgscore), "Low Score: "+str(lowscore), "Mutation Rate: "+str(mut))
            if dropoutrate<0.5:
                dropoutrate+=0.1
    np.save("save.npy",np.array(genes[best]))
    # print("Top Score: "+str(topscore), "Composition: ", genes[best])





    # n=network(genes[best],[128,10,20,2])
    # ob = env.reset()
    # while True:
    #     ob=np.array([ob])
    #     env.render()
    #     ob, reward, done,info=env.step(np.argmax(n.predict(ob,0)))
    #     if info['ale.lives']==4:
    #         break


    # print(ob.size,reward, info)

