import torch as torch
import torch.nn as nn

from torch.nn import init

import torch.utils.model_zoo as model_zoo
import numpy as np


from torchvision.ops import RoIPool, RoIAlign

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def get_idx(batch_size, n_output, device=None):
    idx = torch.arange(float(batch_size), dtype=torch.float, device=device).view(1,
                                                                                 -1)  # 比如batch_size=8,arange(batch_size)=tensor([0,1,2,...,7]),是一维的，view(1,-1)得到tensor([[0,1,2,...,7]])
    idx = idx.repeat(n_output,
                     1, ).t()  # n_output是一个图像被分的块数，比如100，repeat(第一个维度复制n_output次，第二个维度不变)[[0~7],[0~7],...,[0~7]]100行8列,t()是转置，变为8行100列[[0~0],[1~1],...,[7~7]]
    idx = idx.contiguous().view(-1,
                                1)  # contiguous()返回一个内存连续的有相同数据的 tensor,用于view()前,view(-1, 1)变为800行1列，[[0],...[0],[1],...,[1],...,[7]]
    return idx


def get_blockwise_rois(blk_size, img_size=None, batch_size=0):  # 用来得到每个块的位置信息，一个块用左上和右下的两个坐标表示，blk_size块的个数，比如[10,10]
    if img_size is None: img_size = [1, 1]
    y = np.linspace(0, img_size[0],
                    num=blk_size[0] + 1)  # y==>H 序列生成器linspace(起点0，终点1024，个数11) [0 102.4 204.8 .。。1024 ]
    x = np.linspace(0, img_size[1], num=blk_size[1] + 1)  # x==>W [0 204.8 .。。2048 ]
    a = []
    for n in range(len(y) - 1):
        for m in range(len(x) - 1):
            a += [x[m], y[n], x[m + 1], y[n + 1]]  # 从左到右从上到下扫描 [0 0 204.8 102.4 204.8 0 409.6 102.4...]
    a = torch.tensor(a).view(1, -1)  # tensor([[0 0 204.8 102.4 204.8 0 409.6 102.4...]]) (1,400)
    a = a.repeat(1, batch_size).t()  # (400*batch_size,1)
    a = a.contiguous().view(-1,
                            4)  # (100*batch_size,4) tensor([0 0 204.8 102.4],[204.8 0 409.6 102.4],...,[0 0 204.8 102.4],...)
    return a


class RoiADGNet(nn.Module):
    rois = None

    def __init__(self):
        super().__init__()
        self.backbone = ADGNet_AD1().cuda()
        #You Should Modify the PATH with your PreTrained Model
        self.backbone.load_state_dict(torch.load('xxxx.pkl'))
        self.net = self.backbone.dense

        spatial_scale = 1 / 32
        self.model_type = self.__class__.__name__
        self.roi_pool = RoIAlign((5, 5), spatial_scale=spatial_scale, sampling_ratio=4)  # tv里的函数，一张图像输出维度为(10,10)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, im_data):  # im_data (B,W,H)

        feats_all = self.backbone(im_data)
        local_q2 = feats_all['LcQ2']
        feats_q1 = feats_all['LcQ1']
        batch_size = im_data.size(0)  # B
        rois_data = get_blockwise_rois([10, 10], [im_data.size(2), im_data.size(3)], batch_size).float().to(
            im_data.device)  # (100*B,4)
        idx = get_idx(batch_size, rois_data.size(0) // batch_size, im_data.device)  # (B*100,1) 其中100是10*10得到的

        indexed_rois = torch.cat((idx, rois_data), 1)
        features = self.roi_pool(feats_q1, indexed_rois)
        features = self.pool(features).view(features.size(0), -1)
        features = self.net(features)
        local_q1 = features.view(batch_size, 10, 10)
        out = {}
        out['Q1'] = local_q1
        out['Q2'] = local_q2
        out['LcFt'] = feats_all['LcFt']
        out['Att1'] = feats_all['Att1']
        out['Att2'] = feats_all['Att2']

        return out

        pass


# Feature Extraction Block
class FEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down_sample=False):
        super(FEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch * 4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch * 4)
        self.conv2 = nn.Conv2d(in_ch * 4, in_ch * 4, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch * 4)
        self.conv3 = nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.down = nn.MaxPool2d(stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.down_sample:
            out = self.down(out)

        return out


class MultiScaleFeaturesComplex(nn.Module):

    def __init__(self):
        super(MultiScaleFeaturesComplex, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.f1_extractor_01 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.f1_extractor02 = nn.Sequential(
            nn.Conv2d(1024, 64, groups=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.f2_extractor_01 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.f2_extractor02 = nn.Sequential(
            nn.Conv2d(512, 64, groups=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.f3_extractor_01 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.f3_extractor02 = nn.Sequential(
            nn.Conv2d(256, 64, groups=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.f4_extractor = nn.Sequential(
            nn.Conv2d(2048, 320, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, img):
        x = self.res(img)
        f1 = x['f1']
        f2 = x['f2']
        f3 = x['f3']
        f4 = x['f4']

        f1 = self.f1_extractor_01(f1)
        f1c1 = f1.chunk(8, dim=2)
        f1c2 = []
        for xx in (f1c1):
            for yy in (xx.chunk(8, dim=3)):
                f1c2.append(yy)
        f1 = torch.cat(f1c2, dim=1)
        f1 = self.f1_extractor02(f1)
        f2 = self.f2_extractor_01(f2)
        f2c1 = f2.chunk(4, dim=2)
        f2c2 = []
        for xx in (f2c1):
            for yy in (xx.chunk(4, dim=3)):
                f2c2.append(yy)
        f2 = torch.cat(f2c2, dim=1)
        f2 = self.f2_extractor02(f2)
        f3 = self.f3_extractor_01(f3)
        f3c1 = f3.chunk(2, dim=2)
        f3c2 = []
        for xx in (f3c1):
            for yy in (xx.chunk(2, dim=3)):
                f3c2.append(yy)
        f3 = torch.cat(f3c2, dim=1)
        f3 = self.f3_extractor02(f3)
        f4 = self.f4_extractor(f4)
        out = torch.cat((f1, f2, f3, f4), 1)
        out = self.conv(out)

        return out


class MultiScaleFeaturesSimple(nn.Module):
    def __init__(self):
        super(MultiScaleFeaturesSimple, self).__init__()
        self.res = resnet50_backbone(pretrained=True)
        self.f1_conv = self._make_block(in_ch=256, out_ch=256, num_layers=3)
        self.f2_conv = self._make_block(in_ch=512, out_ch=256, num_layers=2)
        self.f3_conv = self._make_block(in_ch=1024, out_ch=256, num_layers=1)
        self.f4_conv = self._make_block(in_ch=2048, out_ch=256, num_layers=0)
        self.conv = nn.Sequential(
            nn.Conv2d(256*4, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 1, padding=(0, 0)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
    def _make_block(self, in_ch, out_ch, num_layers, block):
        layers = []
        for l in range(num_layers):
            layers.append(block(in_ch, out_ch, True))
            in_ch = out_ch
        layers.append(block(in_ch, out_ch, False))
        return nn.Sequential(*layers)
    def forward(self, img):
        x = self.res(img)
        f1 = x['f1']
        f2 = x['f2']
        f3 = x['f3']
        f4 = x['f4']
        f1 = self.f1_conv(f1)
        f2 = self.f1_conv(f2)
        f3 = self.f1_conv(f3)
        f4 = self.f1_conv(f4)
        out = torch.cat((f1, f2, f3, f4), 1)
        out = self.conv(out)

        return out


# ADGNet Implemented by AD#1
class ADGNet_AD1(nn.Module):
    def __init__(self, isComplex=True, isCA = False):
        super(ADGNet_AD1, self).__init__()
        self.isca = isCA
        self.isatt = 1
        self.C = 0.00001
        if isComplex:
            self.msf = MultiScaleFeaturesComplex()
        else:
            self.msf = MultiScaleFeaturesSimple()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ca = nn.Sequential(nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, 512),
                                nn.ReLU(inplace=True)
                                )
        self.att_Q = nn.Sequential(nn.Conv2d(512, 2048, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(2048, 512, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.ReLU(inplace=True),
                                 )
        self.att_S = nn.Sequential(nn.Conv2d(512, 2048, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(2048, 512, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.ReLU(inplace=True),
                                  )


        self.denseQ = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.denseS = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 101),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.msf(img)
        if self.isca:
            ft = self.pool(x)
            ft = ft.view(ft.size(0), -1)
            caweigth = self.ca(ft)
            caweigth = caweigth.view(caweigth.size(0), caweigth.size(1), 1, 1)
            x1 = x * caweigth
        else:
            x1 = x

        if self.isatt == 1:
            attweightQ = self.att_Q(x)
            attweightS = self.att_S(x)

            attweightFTQ = torch.mean(attweightQ, dim=1)
            attweightFTS = torch.mean(attweightS, dim=1)

            xq = x1 * attweightQ
            # LcQ1 and LcQ2 Are Used for Local Quality Map Generation. Not Involved in the Training Stage
            LcQ1 = xq.detach()
            xs = x1 * attweightS
            AdgSim = (2 * attweightFTQ * attweightFTS + self.C) / (
                    attweightFTQ * attweightFTQ + attweightFTS * attweightFTS + self.C)
            LcQ2 = AdgSim.detach()
            AdgSim = AdgSim.view(AdgSim.size(0), -1)
            AdgSim = torch.mean(AdgSim, 1)
        else:
            xq = x1
            xs = x1
        out = {}
        xq = self.pool(xq)
        xs = self.pool(xs)
        xq = xq.view(xq.size(0), -1)
        xs = xs.view(xs.size(0), -1)

        out['Q'] = self.denseQ(xq)
        out['S'] = self.denseS(xs)
        out['E'] = AdgSim
        out['LcQ2'] = LcQ2
        out['LcQ1'] = LcQ1
        out['Att1'] = attweightQ
        out['Att2'] = attweightS

        return out


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f1 = x
        x = self.layer2(x)
        f2 = x
        x = self.layer3(x)
        f3 = x
        x = self.layer4(x)
        f4 = x
        out = {}
        out['f1'] = f1
        out['f2'] = f2
        out['f3'] = f3
        out['f4'] = f4

        return out


def resnet50_backbone(pretrained=False):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
