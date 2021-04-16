# -*- coding: utf-8 -*-
# @Date    : 16-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import torch
import numpy as np
import gc

from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_image_embeddings(net, image_loader):
    net.eval()
    embeds = []
    with torch.no_grad():
        for images in tqdm(image_loader):
            with torch.cuda.amp.autocast():
                images = images.to(device)
                features = net(images)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


def get_text_embeddings(net, text_loader):
    net.eval()
    embeds = []
    with torch.no_grad():
        for text in tqdm(text_loader):
            with torch.cuda.amp.autocast():
                text = text.to(device)
                features = net(input_ids=text)
            text_embeddings = features[0][:, 0, :].detach().cpu().numpy()
            embeds.append(text_embeddings)
    text_embeddings = np.concatenate(embeds)
    print(f'Our text embeddings shape is {text_embeddings.shape}')
    del embeds
    gc.collect()
    return text_embeddings
