import os
import argparse
import random
import numpy as np
import readlabel
import scipy.io as scio
import csv
import data_loader
import torch
from scipy import stats
import numpy as np
import models
import data_loader
import scipy.io as scio

class IQASolver(object):

    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.model_hyper = models.IQANet03().cuda()
        self.model_hyper.train(True)
        self.l1_loss = torch.nn.MSELoss().cuda()
        self.l2_loss = torch.nn.BCELoss().cuda()
        self.l3_loss = torch.nn.L1Loss().cuda()
        backbone_params = list(map(id, self.model_hyper.res.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params,
                                      self.model_hyper.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)
        train_loader = data_loader.DataLoader02(config.dataset, path, train_idx, config.patch_size,
                                                config.train_patch_num, batch_size=config.batch_size, istrain=True)
        test_loader = data_loader.DataLoader02(config.dataset, path, test_idx, config.patch_size, config.test_patch_num,
                                               istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        srcc_per_epoch = np.zeros((1, self.epochs), dtype=float)
        plcc_per_epoch = np.zeros((1, self.epochs), dtype=float)
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTestSRCC02')
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            bcnt = 0
            for img, label, smlabel in self.train_data:
                img = torch.tensor(img.cuda())
                label = torch.tensor(label.cuda())
                smlabel = torch.tensor(smlabel.cuda())
                self.solver.zero_grad()
                self.model_hyper.train(True)
                pred = self.model_hyper(img)
                predQ = pred['Q']
                predS = pred['S']
                predE = pred['E']
                smlabel = smlabel.float().detach()
                pred_scores = pred_scores + predQ.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                Wt = torch.where(smlabel == 0, torch.ones_like(smlabel) * 0.1, torch.ones_like(smlabel) * 1.0)
                lossS = self.l2_loss(predS * Wt, smlabel)
                lossQ = self.l3_loss(predQ.squeeze(), label.float().detach())
                lossE = torch.abs(predE+label.float().detach().squeeze()-1)
                lossE = torch.where(lossE<0.005,torch.ones_like(lossE) * 0.0,lossE)
                lossE = torch.mean(lossE)

                loss = lossQ + lossS * 1.0 +lossE
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
                bcnt+=1

            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            test_srcc, test_plcc, test_srcc02 = self.test(self.test_data, t)
            plcc_per_epoch[0, t] = test_plcc
            srcc_per_epoch[0, t] = test_srcc
            print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_srcc02))

            lr = self.lr
            if t >= 5:
                self.lrratio = 1
            self.paras = [{'params': self.hypernet_params, 'lr': lr * self.lrratio},
                          {'params': self.model_hyper.res.parameters(), 'lr': self.lr}
                          ]
            self.solver = torch.optim.Adam(self.paras, weight_decay=self.weight_decay)

        return srcc_per_epoch, plcc_per_epoch


    def test(self, data, t):
        """Testing"""
        self.model_hyper.train(False)
        self.model_hyper.eval()
        pred_scores = []
        pred_scores02 = []
        gt_scores = []
        cnt = 0
        for img, label, _ in data:
            # Data.
            img = torch.tensor(img.cuda())
            label = torch.tensor(label.cuda())
            pred = self.model_hyper(img)

            predQ = pred['Q']
            predQ2 = pred['E']
            pred_scores02.append(float(predQ2.item()))
            pred_scores.append(float(predQ.item()))
            gt_scores = gt_scores + label.cpu().tolist()
            cnt += 1

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        pred_scores02 = np.mean(np.reshape(np.array(pred_scores02), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_srcc02, _ = stats.spearmanr(pred_scores02, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        self.model_hyper.train(True)

        return test_srcc, test_plcc, test_srcc02

def main(config):
    folder_path='../data'
    idx_test = scio.loadmat('idx_testK02.mat')
    idx_test = idx_test['idx'].astype(int)
    sel_num_new = idx_test[0, :]
    train_index = sel_num_new[0:int(round(0.8 * len(sel_num_new)))]
    test_index = sel_num_new[int(round(0.8 * len(sel_num_new))):len(sel_num_new)]
    solver = IQASolver(config, folder_path, train_index, test_index)
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='koniq-10k',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=1,
                        help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=1,
                        help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=5,
                        help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=6, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224,
                        help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=1, help='Train-test times')

    config = parser.parse_args()
    main(config)

