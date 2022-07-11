import torch
from scipy import stats
import numpy as np
import models
import data_loader
import scipy.io as scio
from myutil import getConfig

class IQASolver(object):
    """Solver for training and testing hyperIQA"""

    def __init__(self, config, path, train_idx, test_idx):

        self.epochs = config.epochs
        self.myconfig = getConfig()
        self.myconfig.device = torch.device("cuda:%s" %self.myconfig.GPU_ID if torch.cuda.is_available() else "cpu")
        self.model_hyper = models.IQANetEnc(self.myconfig).cuda()
        self.model_hyper.train(True)
        self.ls_loss = torch.nn.BCELoss().cuda()
        self.lq_loss = torch.nn.L1Loss().cuda()
        self.batch_size = config.batch_size
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
        train_loader = data_loader.DataLoader02(config.dataset, path, train_idx, batch_size=self.batch_size, istrain=True)
        test_loader = data_loader.DataLoader02(config.dataset, path, test_idx, batch_size= self.batch_size, istrain=False)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

    def train(self):
        """Training"""
        srcc_per_epoch = np.zeros((1, self.epochs), dtype=np.float)
        plcc_per_epoch = np.zeros((1, self.epochs), dtype=np.float)
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC')
        train_enc_inputs = torch.ones(self.batch_size, 12*16 + 1).to(self.myconfig.device)
        for t in range(self.epochs):
            epoch_loss = []
            pred_scores = []
            gt_scores = []
            bcnt = 0
            for img, label, smlabel in self.train_data:
                img = img.clone().detach().cuda()
                label = label.clone().detach().cuda()
                smlabel = smlabel.clone().detach().cuda()

                self.solver.zero_grad()
                self.model_hyper.train(True)
                pred = self.model_hyper(img, train_enc_inputs)
                predQ = pred['Q']
                predS = pred['S']
                predE = pred['E']

                smlabel = smlabel.float().detach()
                pred_scores = pred_scores + predQ.cpu().tolist()
                gt_scores = gt_scores + label.cpu().tolist()
                Wt = torch.where(smlabel == 0, torch.ones_like(smlabel) * 0.1, torch.ones_like(smlabel) * 1.0)
                lossS = self.ls_loss(predS * Wt, smlabel)
                lossQ = self.lq_loss(predQ.squeeze(), label.float().detach())
                lossE = self.lq_loss(predE.squeeze(),label.float().detach())
                lossE = torch.where(lossE < 0.005, torch.ones_like(lossE) * 0.0, lossE)
                lossE = torch.mean(lossE)

                loss = lossQ + lossS * 0.4 + lossE
                epoch_loss.append(loss.item())
                loss.backward()
                self.solver.step()
                bcnt += 1
                #print(bcnt)
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
        test_enc_inputs = torch.ones(self.batch_size, 12*16 + 1).to(self.myconfig.device)
        pred_scores = []
        pred_scores02 = []
        gt_scores = []
        cnt = 0
        for img, label, _ in data:
            # Data.
            torch.cuda.empty_cache()
            img = img.clone().detach().cuda()
            label = label.clone().detach().cuda()
            pred = self.model_hyper(img, test_enc_inputs)

            predQ = pred['Q']
            predQ2 = pred['E']



            pred_scores = pred_scores + predQ.detach().cpu().numpy().tolist()
            pred_scores02 = pred_scores02 + predQ2.detach().cpu().numpy().tolist()
            gt_scores = gt_scores + label.detach().cpu().tolist()
            cnt += 1

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, 1)), axis=1)
        pred_scores02 = np.mean(np.reshape(np.array(pred_scores02), (-1, 1)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, 1)), axis=1)


        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_srcc02, _ = stats.spearmanr(pred_scores02, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        scio.savemat('E:\\ADGNet01\\FFgt%03d.mat' % t, {'gt': gt_scores})
        scio.savemat('E:\\ADGNet01\\FFpred%03d.mat' % t, {'pred': pred_scores})
        scio.savemat('E:\\ADGNet01\\FF2pred%03d.mat' % t, {'pred02': pred_scores02})

        self.model_hyper.train(True)

        return test_srcc, test_plcc, test_srcc02

