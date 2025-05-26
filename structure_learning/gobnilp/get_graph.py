import networkx as nx

def gobnilp(data):

    # empty graph
    graph = nx.DiGraph()
    path = 'structure_learning/gobnilp/result/'+data+'.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)-2, -1, -1):
            if lines[i].startswith('gap '):
                break
            if '<-' in lines[i] and ',' in lines[i]:
                index = lines[i].rfind(',')
                s = lines[i][:index]
                e = s.strip().split('<-')
                target = e[0]
                sources = e[1].split(',')
                for source in sources:
                    graph.add_edge(source, target)
    return graph

if __name__ == '__main__':
    i = 5000
    gobnilp('cancer_'+str(i))
    gobnilp('alarm_'+str(i))
    gobnilp('insurance_'+str(i))
    gobnilp('child_'+str(i))
    gobnilp('water_'+str(i))
    gobnilp('child_'+str(i))
    gobnilp('earthquake_'+str(i))
