# -*- coding: utf-8 -*-
# @Date    : 16-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import pandas as pd
import torch
import numpy as np
import gc

from torch.utils.data import DataLoader

from data import ShopeeDataset
from tqdm import tqdm
from config import args
from models import Net


def get_image_embeddings(net, image_loader):
    net.eval()
    embeds = []
    with torch.no_grad():
        for images in tqdm(image_loader):
            with torch.cuda.amp.autocast():
                images = images.to(args.device)
                features = net(images)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    model = Net()
    data = pd.read_csv(args.train_fold)
    dataset = ShopeeDataset(data=data, root_dir=args.train_dir,
                            transform=args.train_args)
    dataloder = DataLoader(dataset, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=True)
    embeddings = get_image_embeddings(
        model, dataloder
    )
    np.save('E_I_0', embeddings)
