# -*- coding: utf-8 -*-
# @Date    : 11-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0

import os
import time

import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import Dataset

from config import args
from models import ShopeeModel

SERIAL_EXEC = xmp.MpSerialExecutor()
# Only instantiate model weights once in memory.
WRAPPED_MODEL = xmp.MpModelWrapper(ShopeeModel())


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
            return image, torch.tensor(label).long()
        else:
            return image


global args


def train_tpu():
    torch.manual_seed(1)

    def get_dataset():
        fold_number = 0
        train_ = pd.read_csv(args.train_fold)
        train = ShopeeDataset(train_[train_['fold'] != fold_number].reset_index(drop=True))
        test = ShopeeDataset(train_[train_['fold'] != fold_number].reset_index(drop=True), transform=args.test_args)
        return train, test

    # Using the serial executor avoids multiple processes
    # to download the same data.
    train_dataset, test_dataset = SERIAL_EXEC.run(get_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True)

    # Scale learning rate to num cores
    learning_rate = 1e-5 * xm.xrt_world_size()

    # Get loss function, optimizer, and model
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    def train_loop_fn(loader):
        tracker = xm.RateTracker()
        model.train()
        for x, (data, label) in enumerate(loader):
            optimizer.zero_grad()
            output = model(image=data, label=label, get_embedding=args.get_embeddings)
            loss = loss_fn(output, label)
            loss.backward()
            xm.optimizer_step(optimizer)
            tracker.add(args.batch_size)
            if x % 20 == 0:
                print('[xla:{}]({}) Loss={:.5f} Rate={:.2f} GlobalRate={:.2f} Time={}'.format(
                    xm.get_ordinal(), x, loss.item(), tracker.rate(),
                    tracker.global_rate(), time.asctime()), flush=True)

    def test_loop_fn(loader):
        model.eval()
        for x, (data, label) in enumerate(loader):
            output = model(image=data, label=label, get_embedding=args.get_embeddings)
            loss = loss_fn(output, label)
            if x % 20 == 0:
                print('[xla:{}]({}) Loss={:.5f}'.format(xm.get_ordinal(), x, loss.item()), flush=True)
    for epoch in range(1, args.n_epochs + 1):
        para_loader = pl.ParallelLoader(train_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device))
        xm.master_print("Finished training epoch {}".format(epoch))

        para_loader = pl.ParallelLoader(test_loader, [device])
        test_loop_fn(para_loader.per_device_loader(device))


# Start training processes
def _mp_fn(rank, flags):
    global args
    torch.set_default_tensor_type('torch.FloatTensor')
    train_tpu()


xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores,
          start_method='fork')

