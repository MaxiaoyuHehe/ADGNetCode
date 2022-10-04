import torch
import torchvision
import folders

class IQADataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, batch_size=1, istrain=True):

        self.batch_size = batch_size
        self.istrain = istrain

        if dataset == 'koniq-10k':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((384, 512)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])


        if dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms)

    def get_data(self):
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=8, shuffle=False, drop_last=True)
        return dataloader
