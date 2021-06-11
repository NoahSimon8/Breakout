import numpy as np
from random import*
from multiprocessing import Pool
from time import time


class Algorithem:
    def __init__(self,Gensize,Popsize,reward,mutation):
        self.gensize=Gensize
        self.popsize=Popsize
        self.genes=[]
        self.rewards=[]
        self.reward=reward
        self.mutation=mutation

    def testing(self,x):
        print(len(x),x[1].shape)

    def generation(self,prev="None",best=None,mut=0,poolsize=1):
        start = time()

        # print(__name__)
        if prev=="None":
            self.genes=[]
            self.rewards=[]
            for i in range(self.popsize):
                self.genes.append(np.random.rand(self.gensize)*2-1)
            splitgenes = []
            splitter = int(len(self.genes) / poolsize)
            remaindernum=len(self.genes)%splitter
            print("REMAINDER NUM",remaindernum)
            remainders=self.genes[(len(self.genes)-remaindernum):]
            for i in range(len(remainders)):
                remainders[i]={"index":i+(len(self.genes)-remaindernum),"gene":remainders[i]}


            print("REMAINDERS LENGTH",len(remainders))
            for i in range(poolsize):
                splitgenes.append(self.genes[splitter * i : splitter * i + splitter])
                for n in range(len(splitgenes[i])):
                    splitgenes[i][n]={"index":splitter * i+n,"gene":splitgenes[i][n]}

            for i in range(len(remainders)):
                splitgenes[i].append(remainders[i])

            p = Pool(poolsize)
            unlisted = p.map(self.reward, splitgenes)
            self.rewards=[unlisted[0]]
            if len(unlisted)>=2:
                for i in range(len(unlisted)-1):
                    self.rewards[0]+=unlisted[i+1]
            self.rewards=self.rewards[0]

            self.rewards=sorted(self.rewards, key=lambda i: i['index'])
            self.rewards=[i["score"]for i in self.rewards]

            newbest=np.argmin(self.rewards)
        else:
            self.genes=[]
            self.rewards=[]
            for i in range(self.popsize):
                new=[]
                num=np.random.randint(0,len(prev[0])-1)
                other=best
                while other==best:
                    other=randint(0,len(prev)-1)
                a=prev[best]
                b=prev[other]
                for n in range(len(prev[best])):
                    if n>=num:
                        new.append(b[n])
                    else:
                        new.append(a[n])
                #mutation
                new=self.mutation(new,mut)

                self.genes.append(new)

            splitgenes = []
            splitter = int(len(self.genes) / poolsize)
            remaindernum=len(self.genes)%splitter
            remainders=self.genes[(len(self.genes)-remaindernum):]
            for i in range(len(remainders)):
                remainders[i]={"index":i+(len(self.genes)-remaindernum),"gene":remainders[i]}

            for i in range(poolsize):
                splitgenes.append(self.genes[splitter * i:splitter * i + splitter])
                for n in range(len(splitgenes[i])):
                    splitgenes[i][n] = {"index": splitter * i + n, "gene": splitgenes[i][n]}

            for i in range(len(remainders)):
                splitgenes[i].append(remainders[i])

            p = Pool(poolsize)
            unlisted = p.map(self.reward, splitgenes)
            self.rewards = [unlisted[0]]
            if len(unlisted) >= 2:
                for i in range(len(unlisted) - 1):
                    self.rewards[0] += unlisted[i + 1]
            self.rewards = self.rewards[0]

            self.rewards = sorted(self.rewards, key=lambda i: i['index'])

            self.rewards=[i["score"]for i in self.rewards]
            self.genes[np.argmax(self.rewards)]=prev[best] #eletism?
            newbest = np.argmin(self.rewards)
            end=time()
            # print(end-start)
            print(self.rewards)

        topscore=min(self.rewards)
        lowscore=max(self.rewards)
        return  self.genes,newbest,topscore, lowscore

