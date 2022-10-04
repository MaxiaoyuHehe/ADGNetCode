import os
import argparse
import random
import numpy as np
from ADGIQASolver import IQASolver
import readlabel
import scipy.io as scio
import csv
import data_loader
import torch
import models
from scipy import stats
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def test(model, epoch, dataset, save_root= 'E:\\ADGNet01'):
    torch.cuda.empty_cache()
    model.train(False)
    model.eval()
    pred_scores = []
    pred_scores02 = []
    gt_scores = []
    cnt = 0
    for img, label, _ in dataset:
        # Data.
        img = img.clone().detach().cuda()
        label = label.clone().detach().cuda()
        pred = model(img)

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
    scio.savemat(os.path.join(save_root, 'AD1gt%03d.mat' %epoch), {'gt': gt_scores})
    scio.savemat(os.path.join(save_root, 'AD1pred%03d.mat' % epoch), {'pred': pred_scores})
    scio.savemat(os.path.join(save_root, 'AD1pred2%03d.mat' % epoch), {'pred02': pred_scores02})

    model.train(True)

    return test_srcc, test_plcc, test_srcc02

def train(model, loss_q, loss_s, optimEarly, optimLate, epochs, datasetTrain, datasetTest, save_root='E:\\ADGNet01'):
    """Training"""

    print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTestSRCC02')
    for t in range(epochs):
        epoch_loss = []
        pred_scores = []
        gt_scores = []
        if t >=5:
            optim  = optimLate
        else:
            optim = optimEarly

        for img, label, smlabel in datasetTrain:
            img = img.clone().detach().cuda()
            label = label.clone().detach().cuda()
            smlabel = smlabel.clone().detach().cuda()

            optim.zero_grad()
            model.train(True)
            pred = model(img)
            predQ = pred['Q']
            predS = pred['S']
            predE = pred['E']

            smlabel = smlabel.float().detach()
            pred_scores = pred_scores + predQ.cpu().tolist()
            gt_scores = gt_scores + label.cpu().tolist()
            Wt = torch.where(smlabel == 0, torch.ones_like(smlabel) * 0.1, torch.ones_like(smlabel) * 1.0)
            lossS = loss_s(predS * Wt, smlabel)
            lossQ = loss_q(predQ.squeeze(), label.float().detach())
            lossE = loss_q(predE.squeeze(), label.float().detach())
            lossE = torch.where(lossE < 0.005, torch.ones_like(lossE) * 0.0, lossE)
            lossE = torch.mean(lossE)

            loss = lossQ + lossS * 0.4 + lossE
            epoch_loss.append(loss.item())
            loss.backward()
            optim.step()

        torch.save(model.state_dict(), os.path.join(save_root, 'AD1model-%03d.pkl'%t))
        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_srcc, test_plcc, test_srcc02 = test(model, t, datasetTest)

        print('%d\t%4.3f\t\t%4.4f\t\t%4.4f\t\t%4.4f\t\t%4.4f' %
              (t + 1, sum(epoch_loss) / len(epoch_loss), train_srcc, test_srcc, test_plcc, test_srcc02))





def main(config):
    #This is the Train_Test Indexes that acheive roughly medium accuracy in our experiment
    #You can also replace is by:
    #len = 10073(#samples in dataset)
    #tain_test_idx = [xx for xx in range(len)]
    #random.shuffle(train_test_idx)
    tain_test_idx = scio.loadmat('tain_test_idx.mat')['idx'].astype(int)[0, :]
    train_index = tain_test_idx[0:int(round(0.8 * len(tain_test_idx)))]
    test_index = tain_test_idx[int(round(0.8 * len(tain_test_idx))):len(tain_test_idx)]

    cur_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iqa_model = models.ADGNet_AD1(isComplex=True, isCA=False).cuda()
    iqa_model.train(True)
    ls_loss = torch.nn.BCELoss().cuda()
    lq_loss = torch.nn.L1Loss().cuda()
    res_backbone_params = list(map(id, iqa_model.msf.res.parameters()))
    other_params = filter(lambda p: id(p) not in res_backbone_params,
                                  iqa_model.parameters())
    lr, lrratio, weight_decay = config.lr, config.lr_ratio, config.weight_decay

    parasEarly = [{'params': other_params, 'lr': lr * lrratio},
             {'params': iqa_model.msf.res.parameters(), 'lr': lr}
             ]
    parasLate = [{'params': iqa_model.parameters(), 'lr': lr}
                  ]
    optimEarly = torch.optim.Adam(parasEarly, weight_decay=weight_decay)
    optimLate = torch.optim.Adam(parasLate, weight_decay=weight_decay)
    dataset_path = 'E:\\DataBase\\koniq10k_1024x768'

    train_loader = data_loader.IQADataLoader(config.dataset, dataset_path, train_index, batch_size=config.batch_size, istrain=True)
    test_loader = data_loader.IQADataLoader(config.dataset, dataset_path, test_index, batch_size=config.batch_size, istrain=False)

    train(model=iqa_model, loss_q=lq_loss, loss_s=ls_loss, optimEarly=optimEarly, optimLate=optimLate, epochs=config.epochs, datasetTrain=train_loader.get_data(), datasetTest=test_loader.get_data())


    print('End.....')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, default='koniq-10k',
                        help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=5,
                        help='Learning rate ratio')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=12, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help='Epochs for training')
    config = parser.parse_args()
    main(config)
