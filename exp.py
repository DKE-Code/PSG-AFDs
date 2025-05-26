import os

def read_attr(filename):
    with open('data/' + filename + '.csv', 'r') as f:
        attr = f.readline().strip().split(',')
    return attr

def read_ground_truth(filename):
    '''
    read ground truth
    :param : filename
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

def read_bnafd(filename,attr):
    '''
    read bnafd result
    Paramete filename: filename
    Return: fd
    '''
    fd = {}
    if '_0.0' in filename:
        with open("results/bnafd/"+filename + '.txt', 'r') as f:
            f.seek(0)
            for i,line in enumerate(f.readlines()):
                line = line.strip().split(';')
                if attr[int(line[1].strip())] not in fd:
                    fd[attr[int(line[1].strip())]] = []
                fd[attr[int(line[1].strip())]].append([attr[int(x.strip())] for x in line[0].split(',')])
    else:
        with open("results/bnafd/"+filename + '.txt', 'r') as f:
            f.seek(0)
            for i,line in enumerate(f.readlines()):
                line = line.strip().split(';')
                if line[1].strip() not in fd:
                    fd[line[1].strip()] = []
                fd[line[1].strip()].append([x.strip() for x in line[0].split(',')])
    return fd

def read_pyro(filename,attr):
    '''
    read pyro result
    Parameters filename: filename
    Parameters k: topk
    Parameters p: percentage
    Return: fd
    '''
    fd = {}
    # noisy
    if '_0.0' in filename:
        with open("results/pyro/"+filename + '.txt', 'r') as f:
            f.seek(0)
            for i,line in enumerate(f.readlines()):
                line = line.strip().split(';')
                if attr[int(line[1].strip())] not in fd:
                    fd[attr[int(line[1].strip())]] = []
                fd[attr[int(line[1].strip())]].append([attr[int(x.strip())] for x in line[0].split(',')])
    else:
        with open("results/pyro/" + filename + '.txt', 'r') as f:
            f.seek(0)
            for i, line in enumerate(f.readlines()):
                line = line.strip().split(';')
                if line[1].strip() not in fd:
                    fd[line[1].strip()] = []
                fd[line[1].strip()].append([x.strip() for x in line[0].split(',')])
    return fd

def read_smi(filename,k,attr):
    '''
    read smi result
    Parameters
    ----------
    filename
    k : topk
    attr

    Returns
    -------
    smi
    '''
    smi = {}
    with open("results/smi/"+filename + '.txt', 'r') as f:
        for i in range(len(attr)):
            while True:
                line = f.readline().strip()
                if line == '':
                    line = f.readline().strip()
                    target = line
                    target = int(target)
                    break
            line = f.readline()
            smi[attr[target]] = []
            for j in range(k):
                line = f.readline().strip()[:-1].replace('[','').replace(']','').split(',')[:-1]
                smi[attr[i]].append([attr[int(x)] for x in line])
    return smi

def read_rfi(filename,k,attr):
    '''
    read rfi result
    Parameters
    ----------
    filename
    k : topk
    attr

    Returns
    -------
    fd
    '''
    smi = {}
    with open("results/rfi/"+filename + '.txt', 'r') as f:
        for i in range(len(attr)):
            while True:
                line = f.readline().strip()
                if line == '':
                    line = f.readline().strip()
                    target = line
                    break
            line = f.readline()
            smi[attr[i]] = []
            for j in range(k):
                line = f.readline().strip()[:-1].replace('[','').replace(']','').split(',')[:-1]
                smi[attr[i]].append([attr[int(x)] for x in line])
    return smi

def read_fdx(filename,attr):
    fdx = {}
    # noisy
    if '_0.0' in filename:
        with open("results/fdx/"+filename + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('(')[0].split('->')
                if line[1].strip() not in fdx:
                    Y = attr[int(line[1].strip())]
                    fdx[Y] = []
                fdx[Y].append([attr[int(x.strip())] for x in line[0].split(',')])
    else:
        with open("results/fdx/"+filename + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip().split('->')
                if line[1].strip() not in fdx:
                    fdx[line[1].strip()] = []
                fdx[line[1].strip()].append([x.strip() for x in line[0].split(',')])
    return fdx

def read_tane(filename,attr):
    '''
    read tane result
    Parameters
    ----------
    filename
    attr

    Returns
    -------
    fd
    '''
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
                target = attr[int(line[1].strip())-1]
                if target not in tane:
                    tane[target] = []
            tane[target].append([attr[int(x.strip())-1] for x in line[0].strip().split(' ')])
    return tane

def precision(ground_truth, fd, attr):
    '''
    calculate precision
    Parameters
    ----------
    ground_truth
    fd
    attr

    Returns
    -------
    precision
    '''
    if len(fd) == 0:
        return 0
    precision = 0
    all = 0
    for key in attr:
        if key in fd:
            for predict in fd[key]:
                p = 0
                if key in ground_truth:
                    for X in ground_truth[key]:
                        p = max(p, len(set(predict).intersection(set(X))) /  len(predict))
                all += 1
                precision += p
    return precision / all


def recall(ground_truth, fd, attr):
    '''
    calculate recall
    Parameters
    ----------
    ground_truth
    fd
    attr

    Returns
    -------
    recall
    '''
    if len(fd) == 0:
        return 0
    recall = 0
    all = 0
    for key in attr:
        if key in ground_truth:
            for X in ground_truth[key]:
                r = 0
                if key in fd:
                    for predict in fd[key]:
                        r = max(r, len(set(predict).intersection(set(X))) / len(X))
                recall+= r
                all += 1
    return recall / all

def output(ground_truth, fd, attr):
    p = precision(ground_truth, fd, attr)
    r = recall(ground_truth, fd, attr)
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    print(p)
    print(r)
    print(f1)

def exp(filename):

    # move to the current directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # read ground truth
    bn_name = filename.split('_')[0]
    ground_truth = read_ground_truth(bn_name)
    # read attr
    if '_0.0' in filename:
        index_rate = filename.index('_0.0')
        attr = read_attr(filename[:index_rate])
    else:
        attr = read_attr(filename)


    if os.path.exists("results/bnafd/"+filename+".txt"):
        bnafd = read_bnafd(filename,attr)
        print("bnafd")
        # p\r\f1
        output(ground_truth,bnafd,attr)

    if os.path.exists("results/smi/"+filename+".txt"):
        smi = read_smi(filename,1,attr)
        print("smi")
        # p\r\f1
        output(ground_truth,smi,attr)

    if os.path.exists("results/rfi/"+filename+".txt"):
        rfi = read_rfi(filename,1,attr)
        print("rfi")
        # p\r\f1
        output(ground_truth,rfi,attr)

    if os.path.exists("results/fdx/"+filename+".txt"):
        fdx = read_fdx(filename,attr)
        print("fdx")
        # p\r\f1
        output(ground_truth,fdx,attr)

    if os.path.exists("results/pyro/"+filename+".txt"):
        pyro = read_pyro(filename,attr)
        print("pyro")
        # p\r\f1
        output(ground_truth,pyro,attr)

    if os.path.exists("results/tane/"+filename+".txt"):
        tane = read_tane(filename,attr)
        print("tane")
        # p\r\f1
        output(ground_truth,tane,attr)


if __name__ == '__main__':


    exp('cancer_5000')
    exp('earthquake_5000')
    exp('asia_5000')
    exp('insurance_5000')
    exp('water_5000')
    exp('alarm_5000')
    exp('alarm_5000_0.01')
    exp('alarm_5000_0.05')
    exp('cancer_5000_0.01')
    exp('cancer_5000_0.05')


