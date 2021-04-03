# -*- coding: utf-8 -*-
# @Date    : 03-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import albumentations as A


class args:
    neck = "option-D"
    pool = "gem"
    backbone = 'gluon_seresnext101_32x4d'
    p_trainable = False
    embedding_size = 512
    n_classes = 11014
    pretrained_weights = None
    crop_size = 512
    train_args = A.Compose([
        A.SmallestMaxSize(512),
        A.RandomCrop(height=crop_size, width=crop_size, p=1.),
        A.HorizontalFlip(p=0.5),
    ])
    test_args = A.Compose([
        A.SmallestMaxSize(512),
        A.CenterCrop(height=crop_size, width=crop_size, p=1.)
    ])
