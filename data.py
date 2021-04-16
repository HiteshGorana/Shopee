# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import os

import cv2
from torch.utils.data import Dataset


class ShopeeDataset(Dataset):

    def __init__(self, data, root_dir=None,
                 transform=None, tokenizer=None,
                 text=False, max_len=128):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text = text

    def __len__(self):
        return len(self.data)

    def _tokenizer(self, text):
        input_ids = self.tokenizer(text, padding='max_length',
                                   truncation=True, max_length=self.max_len,
                                   return_tensors="pt")['input_ids'][0]
        return input_ids

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.text:
            return self._tokenizer(row['title'])
        else:
            img_path = os.path.join(self.root_dir, row.image)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image
