import numpy as np
import scipy.io as scio


def readsmlb():
    MIdx_f=scio.loadmat('MIdx.mat')
    MIdx=MIdx_f['MIdx'].astype(np.bool)
    with open('label02.txt') as f:
        lines = f.readlines()
        N = len(lines)
    lb_list = []
    for line in lines:
        line = line.replace('\n', '')
        lbs = line.split('\t')
        for lb in lbs:
            lb = lb.strip()
            if lb not in lb_list and lb != "" and lb != " ":
                lb_list.append(lb)
    smlbs = np.zeros((N, len(lb_list)), dtype=np.int)
    cnt = 0
    for line in lines:
        line = line.replace('\n', '')
        lbs = line.split('\t')
        for lb in lbs:
            lb = lb.strip()
            if lb in lb_list:
                smlbs[cnt, lb_list.index(lb)] = 1
        cnt += 1
    smlbs=smlbs[MIdx[:,0],:]
    round01Idx = []
    rount02Idx = [i for i in range(smlbs.shape[0])]
    for i in range(smlbs.shape[0]):
        if smlbs[i, lb_list.index('None')] == 0:
            round01Idx.append(i)
        else:
            smlbs[i, lb_list.index('None')]=0

    return smlbs,rount02Idx,rount02Idx,lb_list


if __name__ == '__main__':
    a,b,c,d=readsmlb()
    print('xxx')
