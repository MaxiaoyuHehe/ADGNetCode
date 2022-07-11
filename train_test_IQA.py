import os
import argparse
import random
import numpy as np
from ADGIQASolver import IQASolver
import readlabel
import scipy.io as scio
import csv
import data_loader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config):
    folder_path = {
        'koniq-10k': 'E:\\ImageDatabase\\KONIQ\\',
    }

    img_num = {
        'koniq-10k': list(range(0, 10073)),
    }
    srcc_all = np.zeros((config.train_test_num, config.epochs), dtype=float)
    plcc_all = np.zeros((config.train_test_num, config.epochs), dtype=float)

    idx_test = scio.loadmat('idx_testK02.mat')
    idx_test = idx_test['idx'].astype(int)

    print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
    for i in range(config.train_test_num):
        print('Round %d' % (i + 1))
        sel_num_new=idx_test[i,:]
        train_index = sel_num_new[0:int(round(0.8 * len(sel_num_new)))]
        test_index = sel_num_new[int(round(0.8 * len(sel_num_new))):len(sel_num_new)]
        solver = IQASolver(config, folder_path[config.dataset], train_index, test_index)
        srcc_all[i, :], plcc_all[i, :] = solver.train()

    print('End.....')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='koniq-10k',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=5,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=24, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs for training')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')

    config = parser.parse_args()
    main(config)
