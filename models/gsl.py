import torch.nn as nn
from layers.gsl_layer import *
from utils.train import id_func
from torch_geometric.nn import GCNConv


# GVAE
class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)  # mu, log sigma: W0共享
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=id_func)  #lambda x: x
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=id_func)
        self.dc = InnerProductDecoder(dropout, act=id_func)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)  # N(0, 1)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


# GAE
class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=id_func)
        self.dc = InnerProductDecoder(dropout, act=id_func)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj)

    def forward(self, x, adj):
        z = self.encode(x, adj)
        return self.dc(z), z, None


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


# GVAE
class MyGVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(MyGVAE, self).__init__()
        self.gc1 = GCNConv(input_feat_dim, hidden_dim1, cached=False)
        self.gc2 = GCNConv(hidden_dim1, hidden_dim2, cached=False)
        self.gc3 = GCNConv(hidden_dim1, hidden_dim2, cached=False)
        self.dc = InnerProductDecoder(dropout, act=id_func)

    def encode(self, x, adj, edge_weight):
        hidden1 = F .relu(self.gc1(x, adj, edge_weight))
        return self.gc2(hidden1, adj, edge_weight), self.gc3(hidden1, adj, edge_weight)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)  # N(0, 1)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj, edge_weight=None):
        mu, logvar = self.encode(x, adj, edge_weight)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


# GAE
class MyGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(MyGAE, self).__init__()
        self.gc1 = GCNConv(input_feat_dim, hidden_dim1, cached=False)
        self.gc2 = GCNConv(hidden_dim1, hidden_dim2, cached=False)
        self.dc = InnerProductDecoder(dropout, act=id_func)

    def encode(self, x, adj, edge_weight):
        hidden1 = F.relu(self.gc1(x, adj, edge_weight))
        return self.gc2(hidden1, adj, edge_weight)

    def forward(self, x, adj, edge_weight=None):
        z = self.encode(x, adj, edge_weight)
        return self.dc(z), z, None

