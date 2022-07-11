import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
from modeltrans import IQARegression
from modeltransdec import IQARegressionDec

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class IQANetEnc(nn.Module):
    def __init__(self, config):
        super(IQANetEnc, self).__init__()


        self.att_disc_calc = IQARegression(config)
        self.isca = 0
        self.isatt = 1
        self.C = 0.00001
        self.res = resnet50_backbone03(0, 0, pretrained=True)
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ca = nn.Sequential(nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, 512),
                                nn.ReLU(inplace=True)
                                )
        self.att = nn.Sequential(nn.Conv2d(512, 2048, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(2048, 512, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.ReLU(inplace=True),
                                 )
        self.att2 = nn.Sequential(nn.Conv2d(512, 2048, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(2048, 512, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.ReLU(inplace=True),
                                  )

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = nn.Sequential(
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

    def forward(self, img, enc_inputs):
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
        x = torch.cat((f1, f2, f3, f4), 1)
        x = self.conv(x)
        if self.isca == 1:
            ft = self.pool(x)
            ft = ft.view(ft.size(0), -1)
            caweigth = self.ca(ft)
            caweigth = caweigth.view(caweigth.size(0), caweigth.size(1), 1, 1)
            x1 = x * caweigth
        else:
            x1 = x
        if self.isatt == 1:
            attweight = self.att(x)
            attweight2 = self.att2(x)
            extraW = self.att_disc_calc(enc_inputs, attweight, attweight2)
            x2 = x1 * attweight
            xs = x1 * attweight2

        else:
            x2 = x1
            xs = x1
        out = {}
        x2 = self.pool2(x2)
        xs = self.pool2(xs)
        x2 = x2.view(x2.size(0), -1)
        xs = xs.view(xs.size(0), -1)
        out['Q'] = self.dense(x2)
        out['S'] = self.denseS(xs)
        out['E'] = extraW
        return out


#Original ADG
class IQANet03(nn.Module):
    def __init__(self):
        super(IQANet03, self).__init__()
        self.isca = 0
        self.isatt = 1
        self.C = 0.00001
        #self.C = 0.0005
        self.res = resnet50_backbone03(0, 0, pretrained=True)
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
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.ca = nn.Sequential(nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, 512),
                                nn.ReLU(inplace=True)
                                )
        self.att = nn.Sequential(nn.Conv2d(512, 2048, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(2048, 512, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.ReLU(inplace=True),
                                 )
        self.att2 = nn.Sequential(nn.Conv2d(512, 2048, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(2048, 512, groups=512, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.ReLU(inplace=True),
                                  )

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.attconv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.dense = nn.Sequential(
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
        x = torch.cat((f1, f2, f3, f4), 1)
        x = self.conv(x)
        if self.isca == 1:
            ft = self.pool(x)
            ft = ft.view(ft.size(0), -1)
            caweigth = self.ca(ft)
            caweigth = caweigth.view(caweigth.size(0), caweigth.size(1), 1, 1)
            x1 = x * caweigth
        else:
            x1 = x
        if self.isatt == 1:
            attweight = self.att(x)
            attweight2 = self.att2(x)

            attweightFT = torch.mean(attweight,dim=1)
            attweightFT2 = torch.mean(attweight2,dim=1)

            x2 = x1 * attweight
            xs = x1 * attweight2
            extraW = (2 * attweightFT * attweightFT2 + self.C) / (
                    attweightFT * attweightFT + attweightFT2 * attweightFT2 + self.C)
            extraW = extraW.view(extraW.size(0),-1)
            extraW = torch.mean(extraW,1)
        else:
            x2 = x1
            xs = x1
        out = {}
        x2 = self.pool2(x2)
        xs = self.pool2(xs)
        x2 = x2.view(x2.size(0), -1)
        xs = xs.view(xs.size(0), -1)

        out['Q'] = self.dense(x2)
        out['S'] = self.denseS(xs)
        out['D1'] = attweightFT
        out['D2'] = attweightFT2
        out['E'] = extraW
        return out


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SelfAttention(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps

        """
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H

        out = self.gamma * self_attetion + x
        return out


class CrossAttention(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_dim, activation=F.relu):
        super(CrossAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        init_conv(self.f)
        init_conv(self.g)
        init_conv(self.h)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps

        """
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(y).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H

        out = self.gamma * self_attetion + x
        return out




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


class ResNetBackbone03(nn.Module):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(ResNetBackbone03, self).__init__()
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



def resnet50_backbone03(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone03(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
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
