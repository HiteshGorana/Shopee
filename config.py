# -*- coding: utf-8 -*-
# @Date    : 29-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=120, p=0.8),
        A.RandomBrightness(limit=(0.09, 0.6), p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])


def get_valid_transforms(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(p=1.0)
    ])


class args:
    neck = "option-D"
    pool = "gem"
    backbone = 'tf_efficientnet_b4'
    embedding_size = 512
    arcface_s = 45
    n_classes = 11014
    pretrained_weights = None
    crop_size = 512
    n_epochs = 10
    normalization = 'simple'
    class_weights_norm = 'batch'
    weight = False
    DEBUG = False
    SEED = 22
    output = '../working/'
    train_dir = '../input/shopee-product-matching/train_images'
    test_dir = '../input/shopee-product-matching/test_images'
    train_fold = '../input/shopee-folds-5/GroupKFold_train_folds.csv'
    test_data = '../input/shopee-product-matching/test.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_embeddings = False
    batch_size = 32
    batch_size_test = 32
    num_workers = 2
    scale = 30
    margin = 0.5
    folds = 5
    scheduler = False
    to_run_folds = [0]
    scheduler_params = dict(lr_start=1e-5, lr_max=1e-5 * batch_size,
                            lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0,
                            lr_decay=0.8)
    train_args = get_train_transforms(crop_size)
    test_args = get_valid_transforms(crop_size)

    model_path = ''
    n_neighbors_max = 50
    n_neighbors_min = 3
    threshold = 4.5
