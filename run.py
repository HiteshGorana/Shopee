# -*- coding: utf-8 -*-
# @Date    : 28-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import args
from data import Shopee, collate_fn
from loss import loss_fn
from models import Net
from train import train_epoch, valid_epoch, GradualWarmupSchedulerV2


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    set_seed(args.SEED)
    os.makedirs(args.output, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')
    train = pd.read_csv(args.train_fold)

    if args.DEBUG:
        train = train.sample(1000).reset_index(drop=True)

    for fold_number in range(args.folds):
        if fold_number in args.to_run_folds:
            data_train_ = Shopee(train[train['fold'] != fold_number].reset_index(drop=True), aug=args.train_args)
            data = DataLoader(data_train_, batch_size=args.batch_size, collate_fn=collate_fn,
                              num_workers=args.num_workers)
            data_valid_ = Shopee(train[train['fold'] == fold_number].reset_index(drop=True), aug=args.test_args)
            data_valid = DataLoader(data_valid_, batch_size=args.batch_size_test, collate_fn=collate_fn,
                                    num_workers=args.num_workers)

            model = Net(args).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            if args.scheduler:
                scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
                scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                            after_scheduler=scheduler_cosine)
            for epoch in range(1, args.n_epochs + 1):
                if args.scheduler:
                    scheduler_warmup.step(epoch - 1)
                _ = train_epoch(model, data, optimizer, loss_fn)
                val_loss, embeddings = valid_epoch(model, data_valid, loss_fn)
                print('#' * 22 + f' EPOCH : {epoch} ' + '#' * 22)
                print(f'EPOCH : {epoch} VALID LOSS : {sum(val_loss) / len(val_loss)}')
                print('#' * (22 * 2 + 11))
                np.save(f'valid_embeddings_{epoch}_fold{fold_number}', embeddings)
                torch.save(model.state_dict(), f'../working/model_{epoch}_fold{fold_number}.pth')
