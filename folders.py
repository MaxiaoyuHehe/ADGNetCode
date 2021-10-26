import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
import readlabel
from openpyxl import load_workbook


class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        smlbs, _, _, tt = readlabel.readsmlb()
    
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = np.array(float(row['MOS_zscore'])).astype(np.float32)
                mos_all.append(mos)
        labels = np.array(mos_all).astype(np.float32)
        labels = labels / np.max(labels)
        labels = 1 - labels
        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                lbs = smlbs[item, :]
                sample.append((os.path.join(root, 'KonIQ', imgname[item]), labels[item], lbs))
        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target,smtarget = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target,smtarget

    def __len__(self):
        length = len(self.samples)
        return length


def getFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == suffix:
            filename.append(i)
    return filename


def getTIDFileName(path, suffix):
    filename = []
    f_list = os.listdir(path)
    for i in f_list:
        if suffix.find(os.path.splitext(i)[1]) != -1:
            filename.append(i[1:3])
    return filename


class WFolder(data.Dataset):

    def __init__(self, root, index):

        p1lb = scipy.io.loadmat(os.path.join('dataW', 'predOnce.mat'))
        lb = scipy.io.loadmat(os.path.join('dataW', 'lables.mat'))
        p1lb = p1lb['p1']
        lb = lb['lb']
        self.lb = np.squeeze(lb)
        self.p1lb = np.squeeze(p1lb)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        data1=scipy.io.loadmat('dataW/W1-%04d.mat'%index)
        data2 = scipy.io.loadmat('dataW/W2-%04d.mat' % index)
        data1=data1['w1']
        data2 = data2['w2']
        lb1=self.p1lb[index]
        lb2=self.lb[index]
        return data1,data2,lb1,lb2

    def __len__(self):
        length = len(list(self.lb))
        return length


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')