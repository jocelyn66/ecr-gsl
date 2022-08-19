import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from utils.name2object import *
import itertools
import torch.nn as nn

from torch import nn
from utils.train import get_norm_of_matrix, normalize_adjacency


class GAEOptimizer(object):

    def __init__(self, args, model, optimizer, use_cuda, adjs, n_edges, pos_weight):
        self.model = model
        self.optimizer = optimizer
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        if args.encoder=='gae':
            loss = {0:self.weighted_loss_function_gae, 1:self.loss_function_gae1, 2:self.loss_function_gae2, 3:self.loss_function_gae3, 4:self.loss_function_gae4}
            self.loss_fn = loss[args.loss_type]
        else:
            print('gvae loss')
            self.loss_fn = self.weighted_loss_function_gvae

        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.valid_freq = args.valid_freq
        self.n_nodes = args.n_nodes
        self.adj_mxs = adjs
        self.n_edges = n_edges
        # self.w_loss = nn.Parameter(torch.ones(args.n_nodes['Train'], args.n_nodes['Train']), requires_grad=False)
        # self.make_mask()
        # w_loss = self.make_cost_mask()
        # print("loss weight sum", torch.sum(w_loss))
        # w_loss = w_loss.to(self.device)
        self.weighted_cost = nn.BCEWithLogitsLoss(weight=self.make_cost_mask(),reduction='sum',pos_weight=pos_weight)

    def make_mask(self):
        self.w_loss = self.w_loss / self.n_nodes['Train']
        for s in ['doc', 'sent', 'entity_coref', 'event_coref']:   #优先级与处理顺序相反
            self.w_loss[self.adj_mxs[s].row, self.adj_mxs[s].col] = 1/self.n_edges[s]

    def make_cost_mask(self):
        # n_edges = self.n_edges - self.n_nodes['Train']
        mask = torch.ones(self.n_nodes['Train'], self.n_nodes['Train']) / (self.n_nodes['Train'] * self.n_nodes['Train'])
        for s in ['doc', 'sent', 'entity_coref', 'event_coref']:   #优先级与处理顺序相反
            n_edge = self.n_edges[s] - self.n_nodes['Train']
            mask[self.adj_mxs[s].row, self.adj_mxs[s].col] = 1/n_edge
        # 对角线:负样本
        print("###1", torch.sum(mask))
        return mask.to(self.device)

    def weighted_loss_function_gvae(self, preds, orig, mu, logvar, norm=1, pos_weight=None, split='Train'):
        """GVAE"""
        # cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        cost = norm * self.weighted_cost(preds, orig)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / self.n_nodes[split] * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD
    
    def loss_function_gvae(self, preds, orig, mu, logvar, norm=1, pos_weight=None, split='Train'):
        """GVAE"""
        # cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        cost = norm * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=pos_weight)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / self.n_nodes[split] * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD

    def loss_function_gvae1(self, preds, orig, mu, logvar, split='Train'):
        ValueError('no inplementation in loss')
        """L = CE + nuclear_norm"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / self.n_nodes[split] * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

        nuclear_norm = torch.linalg.norm(preds, ord='nuc')
        
        return cost + KLD + self.beta * nuclear_norm

    def loss_function_gae(self, preds, orig, mu, logvar, norm=1, pos_weight=None, split='Train'):
        """GAE"""
        cost = norm * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=pos_weight)
        
        return cost

    def weighted_loss_function_gae(self, preds, orig, mu, logvar, norm=1, pos_weight=None, split='Train'):
        """GAE"""
        
        cost = norm * self.weighted_cost(preds, orig)
        
        return cost

    def loss_function_gae1(self, preds, orig, mu, logvar, split='Train'):
        ValueError('no inplementation in loss')
        """L = CE(A, A') + nuclear_norm(A')"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        # _, s, _ = torch.svd(preds)
        # nuclear_norm1 = s.sum()
        nuclear_norm = torch.linalg.norm(preds, ord='nuc')
        # print("nuclear norm/rank = ", nuclear_norm, nuclear_norm1)  # 有误差

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return cost + self.beta*nuclear_norm
    
    def loss_function_gae2(self, preds, orig, mu, logvar, split='Train'):
        ValueError('no inplementation in loss')
        # 1. H W: 对称, 归一化
        """L = CE(A, A') + nuclear(W) + L1(H) + F(A'-W-H)"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        
        W = (self.model.W + self.model.W.T)/2  #对称constraint
        W[W<0] = 0
        H = (self.model.H + self.model.H.T)/2
        H[H<0] = 0
        # D[D<0] = 0
        H_norm = normalize_adjacency(H)
        W_norm = normalize_adjacency(W)
        # D_norm = normalize_adjacency(D)

        # D = preds - W_norm - H_norm  #?
        D = preds - W - H

        # norm_W = get_norm_of_matrix(self.model.W)
        # norm_H = get_norm_of_matrix(self.model.H)
        # norm_D = get_norm_of_matrix(preds-self.model.W-self.model.H)

        nuclear_norm = torch.linalg.norm(W_norm, 'nuc')
        l1_norm = torch.norm(H_norm, p=1)
        f_norm = torch.linalg.norm(D)
        
        # _, s, _ = torch.svd(W_norm)
        # nuclear_norm1 = s.sum()

        w = self.beta * nuclear_norm
        h = self.alpha * l1_norm
        d = self.gamma * f_norm
        print("norms:", nuclear_norm, l1_norm, f_norm)

        print("results: ", w, h, d)
        print("cost:", cost)
        # return cost + self.beta*nuclear_norm + self.alpha * l1_norm + self.gamma * f_norm
        return cost + w + h + d

    def loss_function_gae3(self, preds, orig, mu, logvar, split='Train'):
        ValueError('no inplementation in loss')
        """L = CE(A,A') + nuclear(W) + L1(A'-W)"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        
        W = (self.model.W + self.model.W.T)/2
        W[W<0] = 0
        W_norm = normalize_adjacency(W)
        D = preds - W
        D[D<0] = 0
        D_norm = normalize_adjacency(D)

        nuclear_norm = torch.linalg.norm(W_norm, 'nuc')
        l1_norm = torch.norm(D_norm, p=1)
        print("norms:", nuclear_norm, l1_norm)

        w = self.beta*nuclear_norm
        h = self.alpha*l1_norm
        print("results:", w, h)

        return cost + w + h

    def loss_function_gae4(self, preds, orig, mu, logvar, split='Train'):
        ValueError('no inplementation in loss')
        """L = CE(A,W+H) + nuclear(W) + L1(H)"""
        
        W = (self.model.W + self.model.W.T)/2  #对称constraint
        W[W<0] = 0
        W_norm = normalize_adjacency(W)

        H = (self.model.H + self.model.H.T)/2
        H[H<0] = 0
        H_norm = normalize_adjacency(H)

        nuclear_norm = torch.linalg.norm(W_norm, 'nuc')
        l1_norm = torch.norm(H_norm, p=1)
        print("norms:", nuclear_norm, l1_norm)

        w = self.beta * nuclear_norm
        h = self.alpha * l1_norm
        print("results:", w, h)

        cost = self.norm[split] * F.binary_cross_entropy_with_logits(W + H, orig, pos_weight=self.pos_weight[split])
        print("cost:", cost)
        return cost + w + h

    def epoch(self, features, adjs, orig, norm, pos_weight):
        
        # adj: adj_norm
        # adj_ = torch.tensor(adj, device=self.device)
        # orig_ = torch.tensor(orig, device=self.device)
        recovered, mu, logvar = self.model(features, adjs)

        loss = self.loss_fn(preds=recovered, orig=orig, mu=mu, logvar=logvar, norm=norm, pos_weight=pos_weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("w.grad", self.model.w_adj.grad)
        # print(self.model.w_adj, F.softmax(self.model.w_adj))

        # print("W.grad:", torch.min(self.model.W.grad), torch.max(self.model.W.grad))
        # print("H.grad:", torch.min(self.model.H.grad), torch.max(self.model.H.grad))

        return loss.item(), mu

    def eval(self, features, adjs):
        # adj_ = torch.tensor(adj, device=self.device)
        # orig_ = torch.tensor(orig, device=self.device)

        with torch.no_grad():
            recovered, mu, logvar = self.model.infer(features, adjs)
            # loss = self.loss_fn(preds=recovered, orig=orig_, mu=mu, logvar=logvar, split=split)
        # return loss.item(), mu
        return 0, mu
