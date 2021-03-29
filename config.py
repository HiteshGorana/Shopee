# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import albumentations as A


class args:
    neck = "option-D"
    pool = "gem"
    backbone = 'res2net101_26w_4s'
    p_trainable = False
    embedding_size = 512
    n_classes = 11014
    pretrained_weights = None
    crop_size = 256
    train_args = A.Compose([A.Resize(height=544, width=672, p=1.),
                            A.RandomCrop(height=crop_size, width=crop_size, p=1.),
                            A.HorizontalFlip(p=0.5),
                            ])
