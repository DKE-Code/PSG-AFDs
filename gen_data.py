import shutil
import numpy as np
import pandas as pd
import igraph as ig
from dagma import utils

def create_dag(n, m):
    '''
    gen a DAG to simulate the data
    :param n: attribute number
    :param m: edge number
    :return: DAG
    '''
    g = utils.simulate_dag(n,m,'ER')
    mmbsize = __mmbsize__(g)
    # to ensure the dag is not too complex，as realistic
    while mmbsize > 10:
        g = utils.simulate_dag(n,m,'ER')
        mmbsize = __mmbsize__(g)
    print('mmbsize:', mmbsize)
    return g

def __mmbsize__(g):
    mmbsise = 0
    G = ig.Graph.Weighted_Adjacency(g.tolist())
    for i in range(G.vcount()):
        mmb = set()
        parents = G.neighbors(i, mode=ig.IN)
        children = G.neighbors(i, mode=ig.OUT)
        mmb = mmb.union(set(parents))
        mmb = mmb.union(set(children))
        for c in children:
            mmb = mmb.union(set(G.neighbors(c, mode=ig.IN)))
        size_i = len(mmb)
        if size_i > mmbsise:
            mmbsise = size_i
    return mmbsise

def show_dag(g):
    G = ig.Graph.Weighted_Adjacency(g.tolist())
    ig.plot(G, layout=G.layout('kk'), bbox=(300, 300), target='output.png')

def gen_data(n, m, g):
    '''
    generate data
    :param n:  number of samples
    :param m:  number of values
    :param g:  DAG
    :return:   data
    '''
    X = np.zeros((n, len(g)))
    # print(X.shape)
    G = ig.Graph.Weighted_Adjacency(g.tolist())
    ordered_vertices = G.topological_sorting()
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        if len(parents) == 0:
            X[:, j] = np.random.randint(0, m, n)
        else:
            temp = np.zeros((n, 1))
            for k in range(len(parents)):
                temp += X[:, parents[k]].reshape(n, 1)
            adds = np.random.choice([-2,-1, 1,2,0], size=(n,1), p=[0.05,0.05,0.05,0.05,0.8])
            temp += adds
            X[:, j] = temp[:, 0]
    return X

def get_fd(s,v,a,e,n,t,g,local=False):
    '''
    get fd
    :param s: sample number
    :param v: value number
    :param a: attribute number
    :param e: edge number
    :param n: noise
    :param t: times
    :param g: dag
    :param local: store
    :return: fd
    '''
    fd = [[] for i in range(len(g))]
    G = ig.Graph.Weighted_Adjacency(g.tolist())
    ordered_vertices = G.topological_sorting()
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        fd[j] = parents

    if local:
        filename = 'gen_s%d_v%d_a%d_e%d_n%s_t%s.txt'%(s,v,a,e,n,t)
        __write_fd__(fd, 'results/ground/%s'% filename)
    return fd

def __write_fd__(fd, filename):
    with open(filename, 'w') as f:
        for i in range(len(fd)):
            if len(fd[i]) == 0:
                continue
            for j in range(len(fd[i])):
                f.write(str(fd[i][j]))
                if j != len(fd[i]) - 1:
                    f.write(', ')
            f.write(' ;')
            f.write(str(i))
            f.write('\n')

def noise_data(num_samples, num_values, num_attrs, num_edges, time, noise):
    '''
    noise data
    :param num_samples:
    :param num_values:
    :param num_attrs:
    :param num_edges:
    :param time:
    :param noise:
    :return:
    '''
    name0 = 'gen_s%d_v%d_a%d_e%d_n0_t%s'%(num_samples, num_values, num_attrs, num_edges, time)
    X = pd.read_csv('data/%s.csv'% name0).values
    values = [np.unique(X[:, i]) for i in range(X.shape[1])]
    # gen data
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.random.rand() < noise:
                # modify with noise probability
                temp = np.random.choice(values[j])
                while temp == X[i, j]:
                    temp = np.random.choice(values[j])
                X[i, j] = temp
    # save
    name = 'gen_s%d_v%d_a%d_e%d_n%s_t%s'%(num_samples, num_values, num_attrs, num_edges, noise, time)
    pd.DataFrame(X).to_csv('data/%s.csv'% name, header=[str(i) for i in range(X.shape[1])], index=False, sep=',', float_format='%d')
    # copy
    shutil.copyfile('results/ground/%s.txt'% name0, 'results/ground/%s.txt'% name)


def gen(num_samples = 1000,
    num_values = 5,
    num_attrs = 30,
    num_edges = 30,
    noise = 0,
    time = 0
        ):
    # generate a DAG
    g = create_dag(num_attrs, num_edges)
    # print(ig.Graph.Weighted_Adjacency(g.tolist()))
    # show_dag(g)

    # generate data
    X0 = gen_data(num_samples, num_values, g)
    # print(X0)
    filename = 'gen_s%d_v%d_a%d_e%d_n%s_t%s.csv'%(num_samples, num_values, num_attrs, num_edges, noise, time)
    pd.DataFrame(X0).to_csv('data/%s'% filename, header=[str(i) for i in range(len(g))], index=False, sep=',', float_format='%d')
    get_fd(num_samples, num_values, num_attrs, num_edges, noise, time, g, local=True)


if __name__ == '__main__':
    for s in [5000]:
        for v in [5]:
            for a in [10]:
                for n in [0]:
                    for t in range(5):
                        gen(s,v,a,a,n,t)
                        print('gen_s%d_v%d_a%d_e%d_n%s_t%s'%(s,v,a,a,n,t))

    for s in [5000]:
        for v in [5]:
            for a in [10]:
                for n in [0.01,0.05]:
                    for t in range(5):
                        noise_data(s,v,a,a,t,n)
