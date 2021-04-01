# -*- coding: utf-8 -*-
# @Date    : 28-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from config import args
from data import Shopee, submission_collate_fn
from models import Net
from train import submission_valid
# import numpy as np
# np.load('./ambedding.npy')
if __name__ == '__main__':
    args.pretrained_weights = "../input/baseline-shopee/model_9.pth"
    test = pd.read_csv('../input/shopee-product-matching/test.csv')
    data_test = Shopee(test, aug=args.test_args, test=True)
    data = DataLoader(data_test, batch_size=32, collate_fn=submission_collate_fn, num_workers=2)
    model = Net(args)
    embedding = submission_valid(model, data)
    np.save('embedding', embedding)