import os

def read_ground_truth(filename):
    '''
    read ground truth
    :param filename: filename
    :return: ground truth
    '''
    ground_truth = {}
    with open("results/ground/"+filename + '.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip().split(';')
            if line[1].strip() not in ground_truth:
                ground_truth[line[1].strip()] = []
            ground_truth[line[1].strip()].append([x.strip() for x in line[0].split(',')])
    return ground_truth

def read_bnafd(filename):
    '''
    read bnafd result
    :param filename:
    :return:
    '''
    fd = {}
    with open("results/bnafd/"+filename + '.txt', 'r') as f:
        f.seek(0)
        for i,line in enumerate(f.readlines()):
            line = line.strip().split(';')
            if line[1].strip() not in fd:
                fd[line[1].strip()] = []
            fd[line[1].strip()].append([x.strip() for x in line[0].split(',')])
    return fd


def read_pyro(filename):
    '''
    read pyro result
    Parameter filename: filename
    Return: pyro result
    '''
    fd = {}
    with open("results/pyro/"+filename + '.txt', 'r') as f:
        f.seek(0)
        for i,line in enumerate(f.readlines()):
            line = line.strip().split(';')
            if line[1].strip() not in fd:
                fd[line[1].strip()] = []
            fd[line[1].strip()].append([x.strip() for x in line[0].split(',')])
    return fd
def read_smi(filename):
    '''
    read smi result
    Parameter
    ----------
    filename
    k
    Returns
    -------
    smi
    '''
    smi = {}
    target = -1
    with open("results/smi/"+filename + '.txt', 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            if line.strip() == '[' or line.strip() == ']':
                continue
            if line.strip().isdigit():
                target = line.strip()
                smi[target] = []
                continue
            line = line.strip()[:-1].replace('[', '').replace(']', '').split(',')[:-1]
            smi[target].append([x for x in line])
    return smi

def read_rfi(filename,k=1):
    '''
    read rfi result
    Parameters
    ----------
    filename
    k

    Returns
    -------
    rfi
    '''
    smi = {}
    target = -1
    with open("results/rfi/"+filename + '.txt', 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            if line.strip() == '[' or line.strip() == ']':
                continue
            if line.strip().isdigit():
                target = line.strip()
                smi[target] = []
                continue
            line = line.strip()[:-1].replace('[', '').replace(']', '').split(',')[:-1]
            smi[target].append([x for x in line])
    return smi



# 读fdx的结果
def read_fdx(filename):
    fdx = {}
    with open("results/fdx/"+filename + '.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip().split('->')
            if line[1].strip() not in fdx:
                fdx[line[1].strip()] = []
            fdx[line[1].strip()].append([x.strip() for x in line[0].split(',')])
    return fdx

def read_tane(filename):
    tane = {}
    with open("results/tane/"+filename + '.txt', 'r') as f:
        t = 0
        for line in f.readlines():
            if t < 2 :
                if line.strip() == "======================================================================":
                    t += 1
                continue
            if line.strip() == '':
                continue
            if line.strip() == "======================================================================":
                break
            line = line.strip().split('(')[0].strip().split('->')
            if line[1].strip() not in tane:
                if line[0].strip() == '':
                    continue
                target = str(int(line[1].strip())-1)
                if target not in tane:
                    tane[target] = []
            tane[target].append([str(int(x.strip())-1) for x in line[0].strip().split(' ')])
    return tane

def precision(ground_truth, fd):
    if len(fd) == 0:
        return 0
    precision = 0
    all = 0
    for key in fd.keys():
        for predict in fd[key]:
            p = 0
            if key in ground_truth:
                for X in ground_truth[key]:
                    p = max(p, len(set(predict).intersection(set(X))) /  len(predict))
            all += 1
            precision+= p
    return precision / all

def recall(ground_truth, fd):
    if len(fd) == 0:
        return 0
    recall = 0
    all = 0
    for key in ground_truth.keys():
        for X in ground_truth[key]:
            r = 0
            if key in fd:
                for predict in fd[key]:
                    r = max(r, len(set(predict).intersection(set(X))) / len(X))
            recall+= r
            all += 1
    return recall / all


def exp(name):
    truth = read_ground_truth(name)
    results = {}

    if os.path.exists("results/bnafd/"+name+".txt"):
        fd = read_bnafd(name)
        p = precision(truth,fd)
        r = recall(truth,fd)
        f1 = 2*p*r/(p+r) if p+r != 0 else 0
        results['fd'] = [p,r,f1]

    if os.path.exists("results/fdx/"+name+".txt"):
        fdx = read_fdx(name)
        p = precision(truth,fdx)
        r = recall(truth,fdx)
        f1 = 2*p*r/(p+r) if p+r != 0 else 0
        results['fdx'] = [p,r,f1]

    if os.path.exists("results/smi/" + name + ".txt"):
        smi = read_smi(name)
        p = precision(truth, smi)
        r = recall(truth, smi)
        f1 = 2 * p * r / (p + r) if p + r != 0 else 0
        results['smi'] = [p, r, f1]

    if os.path.exists("results/rfi/"+name+".txt"):
        rfi = read_rfi(name)
        p = precision(truth,rfi)
        r = recall(truth,rfi)
        f1 = 2*p*r/(p+r) if p+r != 0 else 0
        results['rfi'] = [p,r,f1]

    if os.path.exists("results/pyro/"+name+".txt"):
        pyro = read_pyro(name,1,1)
        len_pyro = 0
        for key in pyro:
            len_pyro += len(pyro[key])
        p = precision(truth,pyro)
        r = recall(truth,pyro)
        f1 = 2*p*r/(p+r) if p+r != 0 else 0
        results['pyro'] = [p, r, f1]

    if os.path.exists("results/tane/"+name+".txt"):
        tane = read_tane(name)
        len_tane = 0
        for key in tane:
            len_tane += len(tane[key])
        # p\r\f1
        p = precision(truth,tane)
        r = recall(truth,tane)
        f1 = 2*p*r/(p+r) if p+r != 0 else 0
        results['tane'] = [p, r, f1]

    return results



if __name__ == '__main__':

    for s in [5000]:
        for v in [5]:
            for a in [10]:
                for n in [0]:
                    for t in range(5):
                        name = 'gen_s%d_v%d_a%d_e%d_n%s_t%d' % (s, v, a, a, n, t)
                        print(exp(name))
