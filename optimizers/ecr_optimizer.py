import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from utils.name2object import *

from torch import nn
from utils.train import get_norm_of_matrix, normalize_adjacency


# class ECROptimizer(object):

#     def __init__(self, model, optimizer, valid_freq, batch_size,
#                  regularizer=None, use_cuda=False, dropout=0.):
#         self.model = model
#         self.regularizer = regularizer
#         self.optimizer = optimizer
#         self.batch_size = batch_size
#         self.loss_fn = getattr(self, name2loss(self.model.gsl))
#         # self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
#         self.use_cuda = use_cuda
#         self.device = torch.device("cuda" if self.use_cuda else "cpu")
#         self.valid_freq = valid_freq
#         self.dropout = dropout

#     def calculate_loss(self, examples):
#         pass

#         predictions, factors = self.model(examples)
#         loss = self.loss_fn(predictions)
#         if self.regularizer:
#             loss += self.regularizer.forward(factors)
#         return loss

#     def epoch(self, examples, target, mask):
#         # batch gd

#         actual_examples = examples[torch.randperm(examples.shape[0]), :]
#         with tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
#             bar.set_description(f'train loss')
#             b_begin = 0
#             total_loss = 0.0
#             iter = 0
#             while b_begin < examples.shape[0]:
#                 input_batch = actual_examples[b_begin:b_begin + self.batch_size].cuda()

#                 l = self.calculate_loss(input_batch)
#                 self.optimizer.zero_grad()
#                 l.backward()
#                 self.optimizer.step()

#                 b_begin += self.batch_size

#                 total_loss += l
#                 iter += 1
#                 bar.update(input_batch.shape[0])
#                 bar.set_postfix(loss=f'{l.item():.4f}')
#         total_loss /= iter
#         return total_loss

#     def evaluate(self, test_data, valid_mode=False):
#         pass

#         valid_losses = []
#         valid_loss = None

#         with torch.no_grad():
#             for idx in enumerate(tqdm(len(test_data))):

#                 loss = self.calculate_loss(test_data[idx])
#                 valid_losses.append(loss.item())

#             if valid_losses:
#                 valid_loss = np.mean(valid_losses)

#         return valid_loss


class GAEOptimizer(object):

    def __init__(self, args, model, optimizer, norm, pos_weight, use_cuda):
        self.model = model
        self.optimizer = optimizer
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma

        loss = {0:self.loss_function_gae, 1:self.loss_function_gae1, 2:self.loss_function_gae2, 3:self.loss_function_gae3, 4:self.loss_function_gae4}
        self.loss_fn = loss[args.loss_type]
        # self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.valid_freq = args.valid_freq
        self.n_nodes = args.n_nodes
        self.norm = norm
        # self.norm = torch.tensor([norm])
        self.pos_weight = pos_weight

    def loss_function_gvae(self, preds, orig, mu, logvar, split='Train'):
        """GVAE"""
        cost = self.norm * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / self.n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD

    def loss_function_gvae1(self, preds, orig, mu, logvar, split='Train'):
        """L = CE + nuclear_norm"""
        cost = self.norm * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / self.n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

        nuclear_norm = torch.linalg.norm(preds, ord='nuc')
        
        return cost + KLD + self.beta * nuclear_norm

    def loss_function_gae(self, preds, orig, mu, logvar, split='Train'):
        """GAE"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return cost

    def loss_function_gae1(self, preds, orig, mu, logvar, split='Train'):
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
        # 1. H W: 对称, 归一化
        """L = CE(A, A') + nuclear(W) + L1(H) + F(A'-W-H)"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        
        W = (self.model.W + self.model.W.T)/2  #对称constraint
        W[W<0] = 0
        H = (self.model.H + self.model.H.T)/2
        H[H<0] = 0
        D = preds - W - H
        D[D<0] = 0
        H_norm = normalize_adjacency(H)
        W_norm = normalize_adjacency(W)
        D_norm = normalize_adjacency(D)

        # norm_W = get_norm_of_matrix(self.model.W)
        # norm_H = get_norm_of_matrix(self.model.H)
        # norm_D = get_norm_of_matrix(preds-self.model.W-self.model.H)

        nuclear_norm = torch.linalg.norm(W_norm, 'nuc')
        l1_norm = torch.norm(H_norm, p=1)
        f_norm = torch.linalg.norm(D_norm)
        
        # _, s, _ = torch.svd(W_norm)
        # nuclear_norm1 = s.sum()

        w = self.beta * nuclear_norm
        h = self.alpha * l1_norm
        d = self.gamma * f_norm
        print("norms:", nuclear_norm, l1_norm, f_norm)

        print("results: ", w, h, d)

        # return cost + self.beta*nuclear_norm + self.alpha * l1_norm + self.gamma * f_norm
        return cost + w + h + d

    # def loss_function_gae2(self, preds, orig, mu, logvar, split='Train'):
    #     # 乘norm系数
    #     """L = CE(A, A') + nuclear(W) + L1(H) + F(A'-W-H)"""
    #     cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        
    #     # H_norm = get_normed_matrix(self.model.H)
    #     # W_norm = get_normed_matrix(self.model.W)

    #     norm_W = get_norm_of_matrix(self.model.W)
    #     norm_H = get_norm_of_matrix(self.model.H)
    #     norm_D = get_norm_of_matrix(preds-self.model.W-self.model.H)

    #     print(norm_W, norm_H, norm_D)

    #     _, s, _ = torch.svd(self.model.W)
    #     nuclear_norm1 = s.sum()

    #     nuclear_norm = torch.linalg.norm(self.model.W, 'nuc')
        
    #     l1_norm = torch.norm(self.model.H, p=1)
    #     f_norm = torch.linalg.norm(preds-self.model.W-self.model.H)

    #     w = norm_W * self.beta*nuclear_norm
    #     h = norm_H * self.alpha * l1_norm
    #     d = norm_D * self.gamma * f_norm
    #     print(nuclear_norm, "=", nuclear_norm1, l1_norm, f_norm)

    #     print("nuclear norm:", w)
    #     print("l1 norm", h)
    #     print("f norm", d)

    #     # return cost + self.beta*nuclear_norm + self.alpha * l1_norm + self.gamma * f_norm
    #     return cost + h + w + d

    def loss_function_gae3(self, preds, orig, mu, logvar, split='Train'):
        """L = CE(A,A') + nuclear(W) + L1(A'-W)"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        
        nuclear_norm = torch.linalg.norm(self.model.W, 'nuc')
        l1_norm = torch.norm(preds-self.model.W, p=1)

        return cost + self.beta*nuclear_norm + self.alpha * l1_norm

    def loss_function_gae4(self, preds, orig, mu, logvar, split='Train'):
        """L = CE(A,W+H) + nuclear(W) + L1(H)"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(self.model.W+self.model.H, orig, pos_weight=self.pos_weight[split])
        
        nuclear_norm = torch.linalg.norm(self.model.W, 'nuc')
        l1_norm = torch.norm(self.model.H, p=1)

        return cost + self.beta*nuclear_norm + self.alpha * l1_norm

    def epoch(self, dataset, adj, orig):
        # adj: adj_norm
        adj_ = torch.tensor(adj, device=self.device)
        orig_ = torch.tensor(orig, device=self.device)
        recovered, mu, logvar = self.model(dataset, adj_)

        loss = self.loss_fn(preds=recovered, orig=orig_, mu=mu, logvar=logvar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        print("W.grad:", torch.min(self.model.W.grad), torch.max(self.model.W.grad))
        print("H.grad:", torch.min(self.model.H.grad), torch.max(self.model.H.grad))

        return loss.item(), mu

    def eval(self, dataset, adj, orig, split):
        adj_ = torch.tensor(adj, device=self.device)
        orig_ = torch.tensor(orig, device=self.device)

        with torch.no_grad():
            recovered, mu, logvar = self.model(dataset, adj_)
            # loss = self.loss_fn(preds=recovered, orig=orig_, mu=mu, logvar=logvar, split=split)
        # return loss.item(), mu
        return 0, mu
