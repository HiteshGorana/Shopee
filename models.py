# -*- coding: utf-8 -*-
# @Date    : 16-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Backbone(nn.Module):

    def __init__(self, name='resnet18', pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)

        if 'regnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'csp' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'res' in name:  # works also for resnest
            self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x


class Net(nn.Module):
    def __init__(self, pretrained=True):
        super(Net, self).__init__()
        self.backbone = Backbone('resnet18', pretrained=pretrained)
        self.global_pool = GeM(p_trainable=False)

    def forward(self, image):
        x = self.backbone(image)
        x = self.global_pool(x)
        embed = x[:, :, 0, 0]
        return embed
