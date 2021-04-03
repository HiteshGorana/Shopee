# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import albumentations as A


class args:
    neck = "None"
    pool = "gem"
    backbone = 'efficientnet_b0'
    p_trainable = False
    embedding_size = 512
    n_classes = 11014
    pretrained_weights = None
    crop_size = 512
    n_epochs = 10
    train_args = A.Compose([A.Resize(height=512, width=512, p=1.),
                            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
                            A.HorizontalFlip(p=0.5),
                            ])
    test_args = A.Compose([A.Resize(height=512, width=512, p=1.), A.Resize(height=crop_size, width=crop_size, p=1.)])