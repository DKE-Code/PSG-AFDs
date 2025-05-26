import exp_gendata
import numpy as np


######################
# utils

def dic2exl(dic):
    '''
    dic to array
    Parameters
    ----------
    dic

    Returns
    -------
    array
    '''
    array = np.zeros((len(dic), len(dic[list(dic.keys())[0]])))
    for i, key in enumerate(dic.keys()):
        array[i] = dic[key]
    return array.T


# set the output format
np.set_printoptions(formatter={'float': '{: 0.8f}'.format})

for s in [5000]:
    for v in [5]:
        for a in [10]:
        # for a in [60,65,70,75,80]:
            for n in [0]:
            # for n in [0.01,0.05]:
                # average 5 times
                result1 = {}
                result1['fd'] = np.array([0, 0, 0])
                result1['fdx'] = np.array([0, 0, 0])
                result1['smi'] = np.array([0, 0, 0])
                result1['rfi'] = np.array([0, 0, 0])
                result1['pyro'] = np.array([0, 0, 0])
                result1['tane'] = np.array([0, 0, 0])
                # vars
                result_vars = {}
                result_vars['fd'] = np.array([0, 0, 0])
                result_vars['fdx'] = np.array([0, 0, 0])
                result_vars['smi'] = np.array([0, 0, 0])
                result_vars['rfi'] = np.array([0, 0, 0])
                result_vars['pyro'] = np.array([0, 0, 0])
                result_vars['tane'] = np.array([0, 0, 0])

                for t in range(5):
                    name = 'gen_s%d_v%d_a%d_e%d_n%s_t%d' % (s, v, a, a, n, t)
                    ri = exp_gendata.exp(name)
                    for key in ri.keys():
                        result1[key] = result1[key]+ri[key]
                # average
                for key in result1.keys():
                    result1[key] =  result1[key]/5.0

                # variance
                for t in range(5):

                    name = 'gen_s%d_v%d_a%d_e%d_n%s_t%d' % (s, v, a, a, n, t)
                    ri = exp_gendata.exp(name)
                    for key in ri.keys():
                        result_vars[key] = result_vars[key] + (ri[key]-result1[key])**2
                for key in result_vars.keys():
                    result_vars[key] = result_vars[key]/5.0
                result_vars = dic2exl(result_vars)

                result1 = dic2exl(result1)

                # output
                print(str(result1).replace('[',' ').replace(']',' '))
                # print(str(result_vars).replace('[',' ').replace(']',' '))

