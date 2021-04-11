# -*- coding: utf-8 -*-
# @Date    : 11-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import math
import os
import warnings
try:
    from cuml.neighbors import NearestNeighbors
except:
    from sklearn.neighbors import NearestNeighbors
import cv2
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset
from tqdm import tqdm

from config import args


class ShopeeDataset(Dataset):

    def __init__(self, data, root_dir=args.train_dir, transform=args.train_args, train=True):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        if self.train:
            label = row.target
            return dict(image=image, label=torch.tensor(label).long())
        else:
            return image


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=args.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class ShopeeModel(nn.Module):

    def __init__(
            self,
            n_classes=args.n_classes,
            model_name=args.backbone,
            fc_dim=args.embedding_size,
            margin=args.margin,
            scale=args.scale,
            use_fc=True,
            pretrained=True):

        super(ShopeeModel, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.1)
            self.classifier = nn.Linear(in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            in_features = fc_dim

        self.final = ArcMarginProduct(
            in_features,
            n_classes,
            scale=scale,
            margin=margin,
            easy_margin=False,
            ls_eps=0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label, get_embedding=False):
        features = self.extract_features(image, get_embedding=get_embedding)
        if get_embedding:
            return features
        else:
            logits = self.final(features, label)
            return logits

    def extract_features(self, x, get_embedding):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc and not get_embedding:
            x = self.dropout(x)
            x = self.classifier(x)
            x = self.bn(x)
        return x


def train_fn(model, data_loader, optimizer, scheduler, epoch, device=args.device):
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc="Training epoch: " + str(epoch + 1))

    for t, data in enumerate(tk):
        optimizer.zero_grad()
        for k, v in data.items():
            data[k] = v.to(device)

        output = model(image=data['image'], label=data['label'], get_embedding=args.get_embeddings)
        loss = torch.nn.CrossEntropyLoss()(output, data['label'])
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

        tk.set_postfix({'loss': '%.6f' % float(fin_loss / (t + 1)), 'LR': optimizer.param_groups[0]['lr']})

    scheduler.step()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, epoch, device=args.device):
    model.eval()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc="Validation epoch: " + str(epoch + 1))

    with torch.no_grad():
        for t, data in enumerate(tk):
            for k, v in data.items():
                data[k] = v.to(device)
            output = model(image=data['image'], label=data['label'], get_embedding=args.get_embeddings)
            loss = torch.nn.CrossEntropyLoss()(output, data['label'])
            fin_loss += loss.item()
            tk.set_postfix({'loss': '%.6f' % float(fin_loss / (t + 1))})
        return fin_loss / len(data_loader)


class ShopeeScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_start=5e-6, lr_max=1e-5,
                 lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.4,
                 last_epoch=-1):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(ShopeeScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]
        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1
        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) /
                  self.lr_ramp_ep * self.last_epoch +
                  self.lr_start)
        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay **
                  (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) +
                  self.lr_min)
        return lr
