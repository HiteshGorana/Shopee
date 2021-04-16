# -*- coding: utf-8 -*-
# @Date    : 15-04-2021
# @Author  : Hitesh Gorana
# @Link    : None
# @Version : 0.0
import torch
from tqdm import tqdm


def cos_similarity_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def get_topk_cossim(test_emb, tr_emb, batchsize=64, k=50, device='cuda:0', verbose=True):
    tr_emb = torch.tensor(tr_emb, dtype=torch.float32, device=torch.device(device))
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=torch.device(device))
    vals = []
    inds = []
    for test_batch in tqdm(test_emb.split(batchsize), disable=1 - verbose):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        vals_batch, inds_batch = torch.topk(sim_mat, k=k, dim=1)
        vals += [vals_batch.detach().cpu()]
        inds += [inds_batch.detach().cpu()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    return vals, inds


def get_topk_cossim_sub(test_emb, tr_emb, vals_x, batchsize=64, k=50, device='cuda:0', verbose=True):
    tr_emb = torch.tensor(tr_emb, dtype=torch.float32, device=torch.device(device))
    test_emb = torch.tensor(test_emb, dtype=torch.float32, device=torch.device(device))
    vals_x = torch.tensor(vals_x, dtype=torch.float32, device=torch.device(device))
    vals = []
    inds = []
    for test_batch in tqdm(test_emb.split(batchsize), disable=1 - verbose):
        sim_mat = cos_similarity_matrix(test_batch, tr_emb)
        sim_mat = torch.clamp(sim_mat, 0, 1) - vals_x.repeat(sim_mat.shape[0], 1)
        vals_batch, inds_batch = torch.topk(sim_mat, k=k, dim=1)
        vals += [vals_batch.detach().cpu()]
        inds += [inds_batch.detach().cpu()]
    vals = torch.cat(vals)
    inds = torch.cat(inds)
    return vals, inds