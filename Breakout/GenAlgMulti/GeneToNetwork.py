import time

import numpy as np
import random

class network():
    def __init__(self,gene=[],layers=None):
        self.gene=gene
        self.layerinfo=layers
        self.w=[]
        if gene!=[]:
            self.create()
        else:
            self.numgenes=self.numGenes()
    def create(self):
        w=[]
        layersize=[]
        for n, i in enumerate(self.layerinfo):
            if n==len(self.layerinfo)-1:
                break
            layersize.append(self.layerinfo[n]*self.layerinfo[n+1])
        sum1=0
        sum2=0
        for onecount,oneiterator in enumerate(layersize):
            w.append([])
            sum2=sum1
            for twocount in range(oneiterator):
                # print(n+sum2,"GENE")
                w[onecount].append(self.gene[twocount+sum2])
                sum1+=1


        for i in range(len(w)):
            w[i]=np.array(w[i]).reshape(self.layerinfo[i],self.layerinfo[i+1])
        self.w=w



    def predict(self,inputs,droprate,prev):

        inputs=inputs/255
        # dropout layer below
        # drop=np.array([0 if random.uniform(0,1)<droprate else 1 for i in range(self.insize*self.midsize1)]).reshape(self.insize,self.midsize1)
        # self.w1=self.w1*drop
        # drop=np.array([0 if random.uniform(0,1)<droprate else 1 for i in range(self.midsize1*self.midsize2)]).reshape(self.midsize1,self.midsize2)
        # self.w2=self.w2*drop
        # drop=np.array([0 if random.uniform(0,1)<droprate else 1 for i in range(self.midsize2*self.endsize)]).reshape(self.midsize2,self.endsize)
        # self.w3=self.w3*drop


        inputs=np.array(inputs)
        cur=inputs
        for i in self.w:
            cur=cur.dot(i)
            cur=self.sigmoid(cur)
        return cur



    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def numGenes(self):
        total=0
        for i in range(len(self.layerinfo)-2):
            total+=self.layerinfo[i]*self.layerinfo[i+1]
        return total
