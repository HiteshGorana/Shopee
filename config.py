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
    backbone = 'tf_efficientnet_b4'
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
    train_fold = '../input/shopee-folds-5/GroupKFold_train_folds.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_embeddings = True
    batch_size = 32
    batch_size_test = 32
    num_workers = 2
    scale = 30
    margin = 0.5
    folds = 5
    scheduler = False
    to_run_folds = [0]
    train_args = A.Compose([
        A.Resize(crop_size, crop_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=120, p=0.8),
        A.RandomBrightness(limit=(0.09, 0.6), p=0.5)])
    test_args = A.Compose([A.Resize(crop_size, crop_size, always_apply=True)])
