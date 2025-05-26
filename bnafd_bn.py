import heapq
import itertools
import time
from collections import deque
import numpy as np
import networkx as nx
import pandas as pd
import structure_learning.gobnilp.get_graph as get_graph
import os

##########################################################################################
## utils
##########################################################################################
def structure_pruning(X):
    """
    find Y， Z-{Y}-->Y is an cAFD
    Parameters
    ----------
    Z : set
    Returns
    -------
    Y : set
    """
    global sub_graph

    # Ancestor graph
    G_copy = sub_graph.copy()
    leaves = deque([n for n in G_copy.nodes if G_copy.out_degree[n] == 0])
    while len(leaves) > 0:
        leaf = leaves.popleft()
        if leaf not in X:
            for p in G_copy.predecessors(leaf):
                if G_copy.out_degree[p] == 1:
                    leaves.append(p)
            G_copy.remove_node(leaf)

    # ancestors
    ancestors = set(G_copy.nodes).difference(X)
    an_back = ancestors.copy()

    # Moral graph
    G_copy = nx.moral_graph(G_copy)

    # DEG
    setList = []
    while len(ancestors) > 0:
        a = ancestors.pop()
        neighbors = set(G_copy.neighbors(a))
        connected = set()
        while len(neighbors) > 0:
            b = neighbors.pop()
            if b in X:
                connected.add(b)
            elif b in ancestors:
                ancestors.remove(b)
                neighbors = neighbors.union(set(G_copy.neighbors(b))-(an_back-ancestors))
        setList.append(connected)

    # CI test
    ans = set()
    for x in X:
        Y = X.difference({x}).difference(G_copy.neighbors(x))
        count = 0
        for y in Y:
            for connected in setList:
                if y in connected and x in connected:
                    count += 1
                    break
        if count == len(Y):
            ans.add(x)

    return ans


def count_values_attrs(attrs):
    '''
    This function calculates the cardinality of a set of attributes.
    :param attr: tuple
    :return: int
    '''
    global attr_value
    value = 1
    for attr in attrs:
        value = value * attr_value[attr]
    return value

def next_count(cur_count, index_Y):
    '''
    use n_XY to calculate n_X
    :param cur_count: dict
    :param cur_attrs: tuple
    :param attrs: tuple
    :return: dict
    '''
    next_count = {}
    for key in cur_count.keys():
        key1 = key[:index_Y] + key[index_Y + 1:]
        if key1 in next_count.keys():
            next_count[key1] += cur_count[key]
        else:
            next_count[key1] = cur_count[key]
    return next_count

def next_count_from_all(all_count, attrs):
    '''
    use the all_count to calculate the count of atts
    :param all_count: dict
    :param attrs: tuple
    :return: dict
    '''
    next_count = {}
    for key in all_count.keys():
        key1 = tuple(key[i] for i in attrs)
        if key1 in next_count.keys():
            next_count[key1] += all_count[key]
        else:
            next_count[key1] = all_count[key]
    return next_count

def next_count_from_bottom(bottom_count,bottom_attrs, attrs):
    '''
    use the bottom_count to calculate the count of atts
    :param bottom_count: dict
    :param bottom_attrs: tuple
    :param attrs: tuple
    :return: dict
    '''
    index = [bottom_attrs.index(i) for i in attrs]
    next_count = {}
    for key in bottom_count.keys():
        key1 = tuple(key[i] for i in index)
        if key1 in next_count.keys():
            next_count[key1] += bottom_count[key]
        else:
            next_count[key1] = bottom_count[key]
    return next_count

def smi(x,xy,X,Y,index_Y,N,alpha):
    '''
    smi(X;Y)
    :param x: dict
    :param xy: dict
    :param X: tuple
    :param Y: int
    :param index_Y: int
    :param N: int
    :param alpha: float
    :return: smi(X;Y), upper bound
    '''
    global attr_value
    global attr_count

    y = attr_count[Y]
    Nx = count_values_attrs(X)
    Ny = attr_value[Y]
    k = Nx - len(x)
    nx = np.array(list(x.values()))
    ny = np.array(list(y.values()))
    py_ = (ny+Nx*alpha)/(N+Nx*alpha*Ny)
    hy_ = -np.sum(py_*np.log2(py_))
    hyx2_ = -(k*Ny*alpha)/(N+Nx*alpha*Ny)*np.log2(Ny)
    pxy = np.array([xy[a] for a in xy.keys()])
    leny = {i:0 for i in x.keys()}
    if len(X) == 1:
        if index_Y == 0:
            for a in xy.keys():
                leny[a[1]] += 1
            px = np.array([x[a[1]] for a in xy.keys()])
        else:
            for a in xy.keys():
                leny[a[0]] += 1
            px = np.array([x[a[0]] for a in xy.keys()])
    else:
        for a in xy.keys():
            leny[a[:index_Y]+a[index_Y+1:]] += 1
        px = np.array([x[a[:index_Y]+a[index_Y+1:]] for a in xy.keys()])
    pxyx_ = (pxy+alpha)/(px+Ny*alpha)
    hyx0_ = np.sum((pxy+alpha)*np.log2(pxyx_))/(N+Nx*alpha*Ny)
    sorted_leny = np.array([leny[i] for i in x.keys()])
    hyx1_ = np.sum((Ny-sorted_leny)*np.log2(alpha/(nx+Ny*alpha)))*alpha/(N+Nx*alpha*Ny)
    hyx = np.sum(pxy*np.log2((pxy/px)))
    return hy_+hyx1_+hyx0_+hyx2_, hy_+hyx/(N+Nx*alpha*Ny)

def delete_subset(sub_searches):
    '''
    remove duplicate sets
    :param sub_searches: list
    '''
    global index
    i = 0
    while i < len(sub_searches):
        sub_search = sub_searches[i]
        len_sub_search = len(sub_search)
        father = None
        for attr in sub_search:
            if father == None:
                father = index[attr]
            else:
                father = father.intersection({x for x in index[attr] if len(x) > len_sub_search})
            if len(father) == 0:
                break
        if len(father) != 0:
            sub_searches.remove(sub_search)
        else:
            i += 1

def set2tuple(s):
    '''
    :param s: set
    :return: tuple
    '''
    return tuple(sorted(s))

def is_subset(t1,t2):
    '''
    :param t1: tuple
    :param t2: tuple
    :return: bool
    '''
    i = 0
    for x in t2:
        while i < len(t1):
            if x == t1[i]:
                i += 1
                break
            elif x > t1[i]:
                return False
            else:
                i += 1
        if i == len(t1):
            return False
    return True

def convert_table(DATANAME):
    global header
    df = pd.read_csv("data/%s.csv" % (DATANAME), dtype=str, keep_default_na=False)
    # replace the string with numbers
    for col in df.columns:
        m = df[col].unique()
        for i in range(len(m)):
            df[col] = df[col].replace(m[i], i)
    df.columns = range(len(header))
    df.to_csv("data/convert_table/%s.csv" % (DATANAME), index=False)


##########################################################################################
## class
##########################################################################################
class FD():
    '''
    FD represents a functional dependency in the form X --> Y with a corresponding SMI score.

    Attributes:
    X: A tuple representing the set of attributes on the left side of the functional dependency.
    y: An integer representing the attribute on the right side of the functional dependency.
    score: A float representing the SMI score of the functional dependency.
    '''

    def __init__(self,X,y,score):
        self.score = score
        self.X = X
        self.y = y

    def __str__(self):
        global header
        X = tuple([header[i] for i in self.X])
        Y = header[self.y]
        s = str(set(X))[1:-1]
        return "%s;%s;%f" % (str(s[1:-1].replace("'", "")), Y,self.score)

    # The following methods are used to compare FD objects based on their SMI scores.
    def __gt__(self, other):
        return self.score > other.score
    def __lt__(self, other):
        return self.score < other.score
    def __ge__(self, other):
        return self.score >= other.score
    def __le__(self, other):
        return self.score <= other.score
    def __eq__(self, other):
        return self.score == other.score
    def __ne__(self, other):
        return self.score != other.score


class Finder(object):

    '''
    Find AFDs using bnafd.

    Attributes:
    R: All attributes.
    alpha: A float representing the alpha in SMI score.
    topk: topk to output.
    '''
    def __init__(self, R, alpha, topk):
        self.R = R
        self.alpha = alpha
        self.handled_set = set()
        self.topk = topk

    def run(self):
        '''
        Find AFDs using bnafd.
        '''
        global H
        global N
        global all_count
        global sub_graph
        global index
        global num_attr
        global structure_prune_num
        global score_prune_num
        global judge_num

        # MBs
        sub_searches = []
        index = {}
        for node in self.R:
            # parents
            parents = set(sub_graph.predecessors(node))
            # children
            children = set(sub_graph.successors(node))
            cur = parents.union(children).union({node})
            # children's parents
            for child in children:
                cur = cur.union(set(sub_graph.predecessors(child)))
            cur = frozenset(cur)
            sub_searches.append(cur)
            for attr in cur:
                if attr in index.keys():
                    index[attr].add(cur)
                else:
                    index[attr] = {cur}
        # del duplicate sets
        delete_subset(sub_searches)
        del index

        for sub_search in sub_searches:
            # cAFDs
            HFD = {}
            sub_search = set2tuple(sub_search)
            # all subsets with length > 1
            for i in range(len(sub_search),1,-1):
                for sub_tp in itertools.combinations(sub_search,i):
                    # skip if already handled
                    if sub_tp in self.handled_set:
                        continue
                    # structure pruning
                    Y_set = structure_pruning(set(sub_tp))
                    judge_num += len(sub_tp)
                    structure_prune_num += len(sub_tp)-len(Y_set)

                    for Y in Y_set:
                        if Y in HFD.keys():
                            HFD[Y].append(sub_tp)
                        else:
                            HFD[Y] = [sub_tp]
                    self.handled_set.add(sub_tp)

            # get bottom count
            count_0 = next_count_from_all(all_count, sub_search)

            count_1 = {}
            count_1[sub_search] = count_0
            count_2 = {}
            # search level = l
            for l in range(len(sub_search),1,-1):
                for Y in HFD.keys():
                    while len(HFD[Y]) != 0:
                        XY = HFD[Y][0]
                        # stop if the length of XY is less than l
                        if len(XY) < l:
                            break
                        # generate n_X from n_XY
                        index_Y = XY.index(Y)
                        X = XY[:index_Y]+XY[index_Y+1:]
                        if X not in count_2.keys():
                            if len(X) == 1:
                                count_2[X] = attr_count[X[0]]
                            else:
                                count_2[X] = next_count(count_1[XY], index_Y)
                        # SMI
                        global header_dict
                        score,top = smi(count_2[X],count_1[XY],X,Y,index_Y,N,self.alpha)
                        HFD[Y].pop(0)

                        # branch and bound
                        if len(H[Y]) == self.topk and top < H[Y][0].score:
                            i = 0
                            while i < len(HFD[Y]):
                                set_XY = set(XY)
                                if set(HFD[Y][i]).issubset(set_XY):
                                    score_prune_num += 1
                                    HFD[Y].pop(i)
                                else:
                                    i += 1
                        if len(H[Y]) < self.topk:
                            heapq.heappush(H[Y], FD(X, Y, score))
                        elif score > H[Y][0].score:
                            heapq.heappop(H[Y])
                            heapq.heappush(H[Y], FD(X, Y, score))

                # search level change
                for Y in HFD.keys():
                    i = 0
                    while i < len(HFD[Y]):
                        XY = HFD[Y][i]
                        set_XY = set(XY)
                        if len(XY) < l-1:
                            break
                        else:
                            if XY not in count_2.keys():
                                for l_attrs in count_1.keys():
                                    if set_XY.issubset(set(l_attrs)):
                                        index = [l_attrs.index(attr) for attr in l_attrs if attr not in set_XY]
                                        count_2[XY] = next_count(count_1[l_attrs], index[0])
                                        break
                                if XY not in count_2.keys():
                                        count_2[XY] = next_count_from_bottom(count_0,sub_search ,XY)
                            i += 1
                count_1 = count_2
                count_2 = {}

def main(name,alpha=1, topk=1):
    # fds
    global H
    # sample size
    global N
    # sub_graph
    global sub_graph
    # sample count for each attribute
    global attr_count
    # cardinality of the attribute
    global attr_value
    # sample count for all atts
    global all_count
    # num of atts
    global num_attr
    # table header
    global header
    # num of judges
    global judge_num
    # num of structure prunes
    global structure_prune_num
    # num of score prunes
    global score_prune_num
    # header_dict
    global header_dict

    # init
    score_prune_num = 0
    structure_prune_num = 0
    judge_num = 0
    t0 = time.time()
    DATANAME = name

    # header
    header = pd.read_csv("data/%s.csv" %(name)).columns.to_list()
    # header_dict
    header_dict = {}
    for i in range(len(header)):
        header_dict[header[i]] = i

    # read the table converted to numbers
    if (not os.path.exists("data/convert_table/%s.csv" %(DATANAME))):
        convert_table(DATANAME)
    df = pd.read_csv("data/convert_table/%s.csv" %(DATANAME))
    N = df.shape[0]
    num_attr = len(header)

    # count the sample count for each attribute
    attr_count = [
        df[col].value_counts().to_dict() for col in df.columns
    ]
    # count the cardinality of each attribute
    attr_value = [
        len(attr_count[i]) for i in range(num_attr)
    ]

    # count the sample count for all atts
    all_count = {}
    for i in range(df.shape[0]):
        key = tuple(df.iloc[i])
        if key in all_count.keys():
            all_count[key] += 1
        else:
            all_count[key] = 1

    # get the learned graph
    MODEL = get_graph.gobnilp(DATANAME)
    GRAPH = nx.DiGraph()
    for edge in MODEL.edges:
        GRAPH.add_edge(header_dict[edge[0]], header_dict[edge[1]])

    # Generate an undirected graph, used to calculate connected components
    GRAPHN = nx.Graph()
    for edge in MODEL.edges:
        GRAPHN.add_edge(header_dict[edge[0]], header_dict[edge[1]])
    del df
    cmps = [cmp for cmp in nx.connected_components(GRAPHN)]

    # Initialize the heap
    H = {}
    for i in range(num_attr):
        H[i] = []

    # bnafd
    if len(cmps) == 1:
        sub_graph = GRAPH
        Nodes = [i for i in range(num_attr) if i in sub_graph.nodes]
        finder = Finder(Nodes,alpha,topk)
        finder.run()
    else:
        for cmp in cmps:
            # skip single node
            if len(cmp) == 1:
                continue
            sub_graph = GRAPH.subgraph(cmp)
            finder = Finder(cmp,alpha,topk)
            finder.run()

    t1 = time.time()
    # save the result
    with open("results/bnafd/%s.txt" %(DATANAME), 'w') as f:
        for i in range(len(header)):
            while H[i]:
                fd = heapq.heappop(H[i])
                f.write("%s\n" % str(fd))


    return t1-t0

if __name__ == '__main__':

    # main('asia_5000')
    # main('cancer_5000')
    # main('earthquake_5000')
    # main('insurance_5000')
    # main('water_5000')
    # main('alarm_5000')
    main('alarm_5000_0.01')
    main('alarm_5000_0.05')
    main('cancer_5000_0.01')
    main('cancer_5000_0.05')
