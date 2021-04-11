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
from data import ShopeeDataset
from models import ShopeeModel
from train import train_fn, eval_fn, ShopeeScheduler


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
            data_train_ = ShopeeDataset(train[train['fold'] != fold_number].reset_index(drop=True))
            data = DataLoader(data_train_, batch_size=args.batch_size,
                              num_workers=args.num_workers)
            data_valid_ = ShopeeDataset(train[train['fold'] == fold_number].reset_index(drop=True))
            data_valid = DataLoader(data_valid_, batch_size=args.batch_size_test,
                                    num_workers=args.num_workers)

            model = ShopeeModel().to(args.device)
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.scheduler_params['lr_start'])
            if args.scheduler:
                scheduler = ShopeeScheduler(optimizer, **args.scheduler_params)
            else:
                scheduler = None

            for epoch in range(args.n_epochs):
                model_dir = f'epoch{epoch}_arcface_{args.crop_size}x' \
                            f'{args.crop_size}_{args.backbone}' \
                            f'fold_{fold_number}.pt'
                avg_loss_train = train_fn(model, data, optimizer, scheduler, epoch, args.device)
                avg_loss_valid = eval_fn(model, data_valid, epoch)
                print(
                    f'TRAIN LOSS : {avg_loss_train}  VALIDATION LOSS : {avg_loss_valid}'
                )
                torch.save(model.state_dict(), args.output + model_dir)
                torch.save(dict(epoch=epoch, model_state_dict=model.state_dict(),
                                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),
                           args.output + 'checkpoints_' + model_dir
                           )
