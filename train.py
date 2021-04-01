# -*- coding: utf-8 -*-
# @Date    : 28-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import torch
from tqdm import tqdm
from loss import ArcFaceLoss


def train_epoch(model, data_iter, optimizer, criterion):
    model.train()
    scalar = torch.cuda.amp.GradScaler()
    train_loss = []
    bar = tqdm(data_iter)
    for data, label in bar:
        optimizer.zero_grad()
        # Casts operations to mixed precision
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(ArcFaceLoss(), label, output)

        # Scales the loss, and calls backward()
        # to create scaled gradients
        scalar.scale(loss).backward()

        # Unscales gradients and calls
        # or skips optimizer.step()
        scalar.step(optimizer)

        # Updates the scale for next iteration
        scalar.update()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smooth: %.5f' % (loss_np, smooth_loss))
    return train_loss


def valid_epoch(model, data_iter, criterion, output_=True):
    model.eval()
    embeddings = []
    valid_loss = []
    with torch.no_grad():
        bar = tqdm(data_iter)
        for data, label in bar:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(ArcFaceLoss(), label, output)
            loss_np = loss.detach().cpu().numpy()
            valid_loss.append(loss_np)
            smooth_loss = sum(valid_loss[-100:]) / min(len(valid_loss), 100)
            bar.set_description('loss: %.5f, smooth: %.5f' % (loss_np, smooth_loss))
    if output_:
        return embeddings


def submission_valid(model, data_iter):
    model.eval()
    embeddings = []
    bar = tqdm(data_iter)
    with torch.no_grad():
        for data in bar:
            with torch.cuda.amp.autocast():
                output = model(data, get_embeddings=True)
                embeddings.append(output['embeddings'].detach().cpu())
    embeddings = torch.cat(embeddings).cpu().numpy()
    return embeddings
