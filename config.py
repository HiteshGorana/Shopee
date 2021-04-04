# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import albumentations as A
import torch


class args:
    neck = "option-D"
    pool = "gem"
    backbone = 'tf_efficientnet_b3_ns'
    p_trainable = False
    embedding_size = 512
    n_classes = 11014
    pretrained_weights = None
    crop_size = 512
    n_epochs = 10
    normalization = 'simple'
    class_weights_norm = 'batch'
    weight = False
    DEBUG = False
    SEED = 22
    output = 'working'
    train_fold = '../input/shopee-folds/train_fold.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_embeddings = True
    batch_size = 32
    batch_size_test = 32
    num_workers = 2
    train_args = A.Compose([A.Resize(height=512, width=512, p=1.),
                            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
                            A.HorizontalFlip(p=0.5),
                            ])
    test_args = A.Compose([A.Resize(height=512, width=512, p=1.), A.Resize(height=crop_size, width=crop_size, p=1.)])
