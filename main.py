import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from config import args
from data import Shopee, collate_fn
from loss import loss_fn
from models import Net
from train import train_epoch


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _filter_columns(iterator: pd.DataFrame, string_: str):
    if isinstance(iterator, pd.DataFrame):
        iterator = iterator.columns
    columns_ = []
    for column in iterator:
        if column.startswith(string_):
            columns_.append(column)
    return columns_


def combine_predictions(row: pd.DataFrame, string: str = 'preds', cv: bool = False):
    x = np.concatenate([row[column] for column in _filter_columns(row, string)])
    return np.unique(x) if cv else ' '.join(np.unique(x))


if __name__ == '__main__':
    train = pd.read_csv('../input/shopee-product-matching/train.csv')
    tmp = train.groupby('image_phash').posting_id.agg('unique').to_dict()
    train['oof'] = train.image_phash.map(tmp)
    train['target'] = train['label_group'].factorize()[0]
    train['fold'] = -1
    sgk = GroupKFold(n_splits=2)
    for n, (tdx, vdx) in enumerate(sgk.split(train, train['target'], groups=train['label_group'])):
        train['fold'].iloc[vdx] = n
    data_train_ = Shopee(train[train['fold'] == 0].reset_index(drop=True), aug=args.train_args)
    data = DataLoader(data_train_, batch_size=128, collate_fn=collate_fn, num_workers=2)

    data_valid_ = Shopee(train[train['fold'] == 1].reset_index(drop=True), aug=args.train_args)
    data_valid = DataLoader(data_valid_, batch_size=128, collate_fn=collate_fn, num_workers=2)

    model = Net(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _ = train_epoch(model, data, optimizer, loss_fn)
