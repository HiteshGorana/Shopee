# -*- coding: utf-8 -*-
# @Date    : 11-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import pandas as pd

try:
    from cuml.neighbors import NearestNeighbors
except:
    from sklearn.neighbors import NearestNeighbors
import gc
import torch

from torch.utils.data import DataLoader

from config import args
import numpy as np
from models import ShopeeModel
from data import ShopeeDataset

args.pretrained_weights = False
args.get_embeddings = True


def get_image_neighbors(df, embeddings, threshold=args.threshold):
    n_neighbors = args.n_neighbors_max if len(df) > 3 else args.n_neighbors_min
    model_nearest_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    model_nearest_neighbors.fit(embeddings)
    distances, indices = model_nearest_neighbors.kneighbors(embeddings)
    predictions = []
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
    del model_nearest_neighbors, distances, indices
    gc.collect()
    return predictions


def get_image_embeddings(net, image_loader):
    net.eval()
    embeds = []
    with torch.no_grad():
        for img in image_loader:
            img = img.to(args.device)
            features = net(img, None, args.get_embeddings)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


def combine_predictions(row):
    x = np.concatenate([row['image_predictions']])
    return ' '.join(np.unique(x))


if __name__ == '__main__':
    test = pd.read_csv(args.test_data)
    test_loader = ShopeeDataset(test, root_dir=args.test_dir, transform=args.test_args)
    data = DataLoader(test_loader, batch_size=args.batch_size, num_workers=args.num_workers)
    model = ShopeeModel(pretrained=args.pretrained_weights).to(args.device)
    model.load_state_dict(torch.load(args.model_path))
    embedding = get_image_embeddings(model, data)
    image_predictions = get_image_neighbors(test, embedding)
    test['image_predictions'] = image_predictions
    test['matches'] = test.apply(combine_predictions, axis=1)
    test[['posting_id', 'matches']].to_csv('submission.csv', index=False)
