from turtle import pos
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from utils.name2object import *

from torch import nn, nuclear_norm


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
    def __init__(self):
        pass

    def __init__(self, model, optimizer, beta, n_nodes, norm, pos_weight, valid_freq, use_cuda):
        self.model = model
        self.optimizer = optimizer
        self.beta = beta
        if self.beta>0:
            self.loss_fn = self.loss_function_gvae_nuclear_norm if model.gsl_name=='gave' else self.loss_function_gae_nuclear_norm
        else:
            self.loss_fn = self.loss_function_gvae if model.gsl_name=='gave' else self.loss_function_gae
        # self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.valid_freq = valid_freq
        self.n_nodes = n_nodes
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

    def loss_function_gvae_nuclear_norm(self, preds, orig, mu, logvar, split='Train'):
        """GVAE"""
        cost = self.norm * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / self.n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        _, s, _ = torch.svd(preds)
        nuclear_norm = s.sum()
        print("nuclear norm/rank = ", nuclear_norm)
        
        return cost + KLD + self.beta * nuclear_norm

    def loss_function_gae(self, preds, orig, mu, logvar, split='Train'):
        """GAE"""

        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return cost

    def loss_function_gae_nuclear_norm(self, preds, orig, mu, logvar, split='Train'):
        """GAE"""
        cost = self.norm[split] * F.binary_cross_entropy_with_logits(preds, orig, pos_weight=self.pos_weight[split])
        _, s, _ = torch.svd(preds)
        nuclear_norm = s.sum()
        print("nuclear norm/rank = ", nuclear_norm)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return cost + self.beta*nuclear_norm
    
    def loss_function_gae_nuclear_norm2(self, preds, orig, mu, logvar, split='Train'):
        """GAE"""
        cost = torch.linalg.norm(preds-orig)
        nuclear_norm = torch.linalg.norm(preds, 'nuc')
        print("nuclear norm/rank = ", nuclear_norm)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return cost + self.beta*nuclear_norm

    def epoch(self, dataset, adj, orig):
        adj_ = torch.tensor(adj, device=self.device)
        orig_ = torch.tensor(orig, device=self.device)
        recovered, mu, logvar = self.model(dataset, adj_)

        loss = self.loss_fn(preds=recovered, orig=orig_, mu=mu, logvar=logvar)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), mu

    def eval(self, dataset, adj, orig, split):
        adj_ = torch.tensor(adj, device=self.device)
        orig_ = torch.tensor(orig, device=self.device)

        with torch.no_grad():
            recovered, mu, logvar = self.model(dataset, adj_)
            loss = self.loss_fn(preds=recovered, orig=orig_, mu=mu, logvar=logvar, split=split)

        return loss.item(), mu
