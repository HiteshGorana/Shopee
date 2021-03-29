import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from models import Net
from train import train_epoch, valid_epoch
from data import Shopee, collate_fn
from loss import loss_fn
from config import args


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

    data_ = Shopee(train, aug=args.train_args)
    data = DataLoader(data_, batch_size=3, collate_fn=collate_fn)
    model = Net(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _ = train_epoch(model, data, optimizer, loss_fn)
