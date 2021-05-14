from Cartpole.Faster.GeneticAlgorithemFast import*
gensize=100
popsize=1000
x=np.linspace(1,10,gensize)
y=np.sin(x)



def reward(gene):
    rewards=[]
    for i in gene:

        rewards.append(sum(abs(y - i)))
    return rewards

def mutation(new,rate):
    for n, j in enumerate(new):
        if uniform(0, 1) <= rate:
            new[n] = uniform(-1, 1)
    return new
alg=Algorithem(gensize,popsize,reward,mutation)

best,gen=alg.generation()

for i in range(10000):
    best,gen=alg.generation(gen,best,0.0005)
    # print(alg.reward(gen[best]))