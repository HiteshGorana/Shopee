# -*- coding: utf-8 -*-
# @Date    : 28-03-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from tqdm import tqdm

from loss import ArcFaceLoss


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]


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
