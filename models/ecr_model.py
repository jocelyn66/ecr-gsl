import torch
import torch.nn as nn
from utils.name2object import name2gsl, name2init
import models.gsl as gsls
import tqdm
import numpy as np
from utils.train import preprocess_graph, preprocess_adjacency, id_func
import scipy.sparse as sp


class ECRModel_fine_tune(nn.Module):

    def __init__(self, args, tokenizer, plm_model, schema_list, orig_adj=None):
        super(ECRModel, self).__init__()

        ValueError('wrong implementation in ecr model with fine tune')
        # bert
        self.tokenizer = tokenizer
        self.bert_encoder = plm_model
        self.doc_schema = schema_list[0]
        self.event_schema = schema_list[1]
        self.entity_schema = schema_list[2]

        self.gsl = getattr(gsls, name2gsl[args.encoder])(args.feat_dim, args.hidden1, args.hidden2, args.dropout)
        self.gsl_name = args.encoder
        self.device = args.device
        
        # regularization
        self.loss_type = args.loss_type
        if self.loss_type in [2,3,4]:

            init_adj = self.eye_init(args.n_nodes['Train'])

            init_adj = preprocess_adjacency(init_adj)
            self.W = nn.Parameter(torch.tensor(init_adj), requires_grad=True)

        if self.loss_type in [2, 4]:

            init_adj = self.eye_init(args.n_nodes['Train'])

            init_adj = preprocess_adjacency(init_adj)
            self.H = nn.Parameter(torch.tensor(init_adj), requires_grad=True)

    def rand_init(self, n_nodes, rand_rate=0.1):
        # 随机+对称
        mat = np.random.rand(n_nodes, n_nodes)
        mat = np.where(mat<rand_rate, 1, 0)
        mat = mat + np.eye(n_nodes)
        return (mat + mat.T)/2

    def eye_init(self, n_nodes):
        return np.eye(n_nodes)

    def forward(self, dataset, adj):
        # 待优化
        features = []  # ts list

        # loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32)

        # 遍历句子构造句子子图, 同时记录句子文档id
        for _, sent in enumerate(dataset):

            # batch_size, seq_len = sent['input_ids'].shape

            input_ids = torch.tensor(sent['input_ids'], device=self.device).reshape(1, -1)
            encoder_output = self.bert_encoder(input_ids)
            encoder_hidden_state = encoder_output['last_hidden_state']  # (n_sent, n_tokens, feat_dim)

            masks = []
            # token_masks = torch.tensor(sent['input_mask'])
            token_masks = torch.eye(input_ids.shape[1], device=self.device)

            for _, event in enumerate(sent['output_event']):
                this_mask = token_masks[event['tokens_number']]
                this_mask = torch.mean(this_mask, dim=0, keepdim=True)
                masks.append(this_mask)
            for _,entity in enumerate(sent['output_entity']):
                this_mask = torch.mean(token_masks[entity['tokens_number']], dim=0, keepdim=True)
                masks.append(this_mask)
                
            masks = torch.cat(masks, dim=0).cuda()
            encoder_hidden_state = encoder_hidden_state.squeeze()
            
            features.append(masks @ encoder_hidden_state)

        features = torch.cat(features)    # encoder_hidden_state * input_mask = 所求表征
        print(features)
        return self.gsl(features, adj)  # gae
    

class ECRModel(nn.Module):

    def __init__(self, args):
        super(ECRModel, self).__init__()

        self.gsl = getattr(gsls, name2gsl[args.encoder])(args.feat_dim, args.hidden1, args.hidden2, args.dropout)
        self.gsl_name = args.encoder
        self.device = args.device

        #comb edges
        self.w_adj = nn.Parameter(torch.zeros(4), requires_grad=True)
        self.w_act = nn.Softmax(dim=0)

        # self.w_adj = nn.Parameter(torch.tensor([1.,1.,2.,2.]), requires_grad=False)  #?
        # self.w_act = id_func

        # regularization
        self.loss_type = args.loss_type
        if self.loss_type in [2,3,4]:

            init_adj = self.eye_init(args.n_nodes['Train'])

            init_adj = preprocess_adjacency(init_adj)
            self.W = nn.Parameter(torch.tensor(init_adj), requires_grad=True)

        if self.loss_type in [2, 4]:

            init_adj = self.eye_init(args.n_nodes['Train'])

            init_adj = preprocess_adjacency(init_adj)
            self.H = nn.Parameter(torch.tensor(init_adj), requires_grad=True)

    def rand_init(self, n_nodes, rand_rate=0.1):
        # 随机+对称
        mat = np.random.rand(n_nodes, n_nodes)
        mat = np.where(mat<rand_rate, 1, 0)
        mat = mat + np.eye(n_nodes)
        return (mat + mat.T)/2

    def eye_init(self, n_nodes):
        return np.eye(n_nodes)

    def forward(self, features, adjs):
        # dense -> 加权-> tsp -> forward
        #features:fearture_list['Train']
        #adj:dataset.adjecancy['Train'], sp arr
        w_adj = self.w_act(self.w_adj)
        print(self.w_adj)
        print(w_adj)

        comb_adj = torch.zeros(adjs['sent'].shape, device=self.device)
        # comb_adj = (w_adj[0] * adjs['sent']).tocoo()
        for i, s in enumerate(['sent', 'doc', 'event_coref', 'entity_coref']):
            adj = torch.tensor(adjs[s].toarray(), device=self.device)
            comb_adj += w_adj[i].repeat(adjs[s].shape) * adj
        comb_adj = comb_adj.to_sparse().requires_grad_(True)
        # comb_adj = preprocess_graph(comb_adj)
        # comb_adj = comb_adj.to(self.device)
        return self.gsl(features, comb_adj.coalesce().indices(), comb_adj.coalesce().values())  # comb_adj.coalesce().indices()无梯度
    
    def infer(self, features, adjs):
        #features:fearture_list['Dev']
        #adj:dataset.adjecancy['Dev']
        w_adj = self.w_act(self.w_adj)
        w_sum = w_adj[0] + w_adj[1]
        w_adj = w_adj / w_sum.item()

        comb_adj = torch.zeros(adjs['sent'].shape, device=self.device)
        for i, s in enumerate(['sent', 'doc']):
            adj = torch.tensor(adjs[s].toarray(), device=self.device)
            comb_adj += w_adj[i].repeat(adjs[s].shape) * adj
        comb_adj = comb_adj.to_sparse()
        # comb_adj = preprocess_graph(comb_adj)
        # comb_adj = comb_adj.to(self.device)
        return self.gsl(features, comb_adj.indices(), comb_adj.values())
    