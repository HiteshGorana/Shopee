# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
from torch.utils.data import Dataset
import torch
import numpy as np
import cv2
from config import args


def collate_fn(batch):
    input_dict = {}
    target_dict = {}
    for key in ['input']:
        input_dict[key] = torch.stack([b[key] for b in batch]).to(args.device)
    for key in ['idx']:
        input_dict[key] = torch.stack([b[key] for b in batch]).long()
    for key in ['target']:
        target_dict[key] = torch.stack([b[key] for b in batch]).to(args.device).long()
    return input_dict, target_dict


def submission_collate_fn(batch):
    input_dict = {}
    for key in ['input']:
        input_dict[key] = torch.stack([b[key] for b in batch]).to(args.device)
    for key in ['idx']:
        input_dict[key] = torch.stack([b[key] for b in batch]).long()
    return input_dict,


class Shopee(Dataset):
    def __init__(self, df, aug=None, normalization=args.normalization, img_size=args.crop_size, test=False):
        self.df = df
        self.aug = aug
        self.normalization = normalization
        self.img_size = img_size
        self.test = test
        if self.test:
            self.path = '../input/shopee-product-matching/test_images/'
        else:
            self.path = '../input/shopee-product-matching/train_images/'
        self.eps = 1e-6

    def __getitem__(self, idx):
        image_path = self.path + '/' + self.df.iloc[idx].image
        try:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            _ = e
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.int8)
        if self.aug:
            img = self.augment(img)
        img = img.astype(np.float32)
        if self.normalization:
            img = self.normalize_img(img)
        if not self.test:
            target = torch.tensor(self.df.iloc[idx].target)
        tensor = self.to_torch_tensor(img)
        if self.test:
            feature_dict = dict(idx=torch.tensor(idx).long(), input=tensor)
        else:
            feature_dict = dict(idx=torch.tensor(idx).long(), input=tensor, target=target.float())
        return feature_dict

    def augment(self, img):
        img_aug = self.aug(image=img)['image']
        return img_aug.astype(np.float32)

    def normalize_img(self, img):
        if self.normalization == 'channel':
            pixel_mean = img.mean((0, 1))
            pixel_std = img.std((0, 1)) + self.eps
            img = (img - pixel_mean[None, None, :]) / pixel_std[None, None, :]
            img = img.clip(-20, 20)
        elif self.normalization == 'channel_mean':
            pixel_mean = img.mean((0, 1))
            img = (img - pixel_mean[None, None, :])
            img = img.clip(-20, 20)
        elif self.normalization == 'image':
            img = (img - img.mean()) / img.std() + self.eps
            img = img.clip(-20, 20)
        elif self.normalization == 'simple':
            img = img / 255
        elif self.normalization == 'inception':
            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img / 255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
        elif self.normalization == 'imagenet':
            mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
            std = np.array([58.395, 57.120, 57.375], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)
        else:
            pass
        return img

    def __len__(self):
        return len(self.df)

    @staticmethod
    def to_torch_tensor(img):
        return torch.from_numpy(img.transpose((2, 0, 1)))
