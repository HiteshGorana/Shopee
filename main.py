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

    if args.weight:
        val_counts = train.target.value_counts().sort_index().values
        class_weights = 1 / np.log1p(val_counts)
        class_weights = (class_weights / class_weights.sum()) * args.n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    data_train_ = Shopee(train[train['fold'] == 0].reset_index(drop=True), aug=args.train_args)
    data = DataLoader(data_train_, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers)

    data_valid_ = Shopee(train[train['fold'] == 1].reset_index(drop=True), aug=args.test_args)
    data_valid = DataLoader(data_valid_, batch_size=args.batch_size_test, collate_fn=collate_fn,
                            num_workers=args.num_workers)

    model = Net(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=scheduler_cosine)
    for epoch in range(1, args.n_epochs):
        scheduler_warmup.step(epoch - 1)
        _ = train_epoch(model, data, optimizer, loss_fn)
        val_loss, embeddings = valid_epoch(model, data_valid, loss_fn)
        np.save(f'valid_embeddings_{epoch}', embeddings)
        print('#' * 22 + f' EPOCH : {epoch} ' + '#' * 22)
        print(f'EPOCH : {epoch} VALID LOSS : {sum(val_loss) / len(val_loss)}')
        print('#' * (22 * 2 + 11))
        torch.save(model.state_dict(), f'../working/model_{epoch}.pth')
