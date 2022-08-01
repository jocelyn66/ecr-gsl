import torch
import torch.nn as nn
from utils.name2object import name2gsl, name2init
import models.gsl as gsls
import tqdm


class ECRModel(nn.Module):

    def __init__(self, args, tokenizer, plm_model, schema_list, orig_adj=None):
        super(ECRModel, self).__init__()

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
            self.W = nn.Parameter(torch.tensor(orig_adj / 2), requires_grad=True)
            # self.W = nn.Parameter(torch.Tensor(args.n_nodes['Train'], args.n_nodes['Train']), requires_grad=True)
            # torch.nn.init.xavier_uniform_(self.W)

        if self.loss_type in [2, 4]:
            self.H = nn.Parameter(torch.tensor(orig_adj / 2), requires_grad=True)
            # self.H = nn.Parameter(torch.Tensor(args.n_nodes['Train'], args.n_nodes['Train']), requires_grad=True)
            # torch.nn.init.xavier_uniform_(self.H)

    def forward(self, dataset, adj):
        # features: (feat_dim, n_nodes)
        # 待优化
        features = []  # ts list
        print("####@model", self.W)

        # 遍历句子构造句子子图, 同时记录句子文档id
        for _, sent in enumerate(dataset):

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
        return self.gsl(features, adj)  # gae
    