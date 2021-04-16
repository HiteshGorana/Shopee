# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import os

import cv2
import torch
from torch.utils.data import Dataset

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
