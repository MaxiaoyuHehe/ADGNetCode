import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
import readlabel

class Koniq_10kFolder(data.Dataset):

    def __init__(self, root, index, transform):
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
            lbs = smlbs[item, :]
            sample.append((os.path.join(root, '1024x768', imgname[item]), labels[item], lbs))
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



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')