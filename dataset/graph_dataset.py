import itertools
import numpy as np
import json

import numpy as np
from utils.train import add_new_item

DATA_PATH = './data/'

n_events = {'Train': 3808,'Dev': 1245, 'Test': 1780}
n_entities = {'Train': 4758,'Dev': 1476, 'Test': 2055}


class GDataset(object):

    def __init__(self, args):

        self.name = args.dataset
        self.event_idx = {'Train':[], 'Dev':[], 'Test':[]}
        self.event_chain_dict = {'Train':{}, 'Dev':{}, 'Test':{}}
        self.entity_chain_dict = {'Train':{}, 'Dev':{}, 'Test':{}}
        self.adjacency = {}  # 邻接矩阵, 节点:event mention(trigger), entity mention, 边:0./1.,对角线0,句子关系,文档关系
        self.event_coref_adj = {}  # 节点:event mention, 边: 共指关系(成立:1), 对角线1(但不用作label)
        self.entity_coref_adj = {}
        self.n_nodes = {}
        
        self.rand_node_rate = args.rand_node_rate
        self.n_events = n_events
        self.n_entities = n_entities

        for split in ['Train', 'Dev', 'Test']:
            assert(self.n_events[split] > 0)
            assert(self.n_entities[split] > 0)
            self.n_nodes[split] = self.n_events[split] + self.n_entities[split]

        file = {'Train': args.train_file, 'Dev': args.dev_file,'Test': args.test_file}
        # self.event_coref_adj['Train'] = self.get_event_coref_adj('Train')
        for split in ['Train', 'Dev', 'Test']:
            self.adjacency[split] = self.get_adjacency(file[split], split)
            self.event_coref_adj[split] = self.get_event_coref_adj(split)  # bool矩阵, 对角线0
            entity_idx = list(set(range(self.n_nodes[split])) - set(self.event_idx[split]))
            self.entity_coref_adj[split] = self.get_coref_sub_adj(self.entity_chain_dict[split], entity_idx, split)

    def get_schema(self, path, split=''):
        # chain的schema, item：(chain descrip, id)
        # return: event_schema, entity_schema
        if not split:
            ValueError
        with open(path, 'r') as x:
            schema = json.load(x)
        return schema[1], schema[2]

    def get_adjacency(self, path, split):
        # 构图：
        # 节点：event, entity
        # 边：句子 文档关系
        # 【对角线：0】

        adj = np.zeros((self.n_nodes[split], self.n_nodes[split]))
        last_doc_id = ''
        doc_node_idx = []
        sent_node_idx = []
        cur_idx = -1    # 从0开始顺序处理每个句子，对event chain, entity chain中的mention编号，根据mention出现的顺序
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        for _, line in enumerate(lines):
            sent = json.loads(line)

            #  同一文档rand_rate的概率随机放点
            if last_doc_id != sent['doc_id']:

                num = int(len(doc_node_idx)*self.rand_node_rate)
                if doc_node_idx and num>0:
                    idx = doc_node_idx
                    np.random.shuffle(idx)
                    rand_rows_idx = idx[:num]
                    np.random.shuffle(idx)
                    rand_cols_idx = idx[:num]

                    adj[rand_rows_idx, rand_cols_idx] = 1

                last_doc_id = sent['doc_id']
                doc_node_idx = []
            
            # event mentions
            for _, event in enumerate(sent['event_coref']):
                cur_idx += 1
                sent_node_idx.append(cur_idx)
                self.event_idx[split].append(cur_idx)
                add_new_item(self.event_chain_dict[split], event['coref_chain'], cur_idx)

            # eneity mentions
            for _, entity in enumerate(sent['entity_coref']):
                cur_idx += 1
                sent_node_idx.append(cur_idx)
                add_new_item(self.entity_chain_dict[split], entity['coref_chain'], cur_idx)
            
            # 句子子图
            adj[sent_node_idx[0]:sent_node_idx[-1]+1, sent_node_idx[0]:sent_node_idx[-1]+1] = 1

            doc_node_idx.extend(sent_node_idx)
            sent_node_idx = []

        # constraint: 对称，对角线0
        adj = np.where((adj + adj.T)>0, 1., 0.)
        adj[np.diag_indices_from(adj)] = 1
        adj[np.diag_indices_from(adj)] = 0
        # assert(adj.diagonal(offset=0, axis1=0, axis2=1).all()==0)

        return adj
        
    def get_event_node_idx(self, descrip):
        return int(self.schema_event[descrip])

    def get_entity_node_idx(self, descrip):
        return int(self.schema_entity[descrip]) + self.n_events[descrip]

    def get_event_coref_adj(self, split):
        # event coref关系bool矩阵【对角线：1】
        #  for key in event chain dict:
        adj = np.zeros((self.n_nodes[split], self.n_nodes[split]))
        for key in self.event_chain_dict[split]:
            events = self.event_chain_dict[split][key]

            mask = itertools.product(events, events)
            rows, cols = zip(*mask)
            adj[rows, cols] = 1
        # adj = adj + adj.T   # 处理成对称矩阵

        return ((adj + adj.T)>0)[self.event_idx[split], :][:, self.event_idx[split]]

    def get_coref_sub_adj(self, dict, idx, split):
        # event coref关系bool矩阵【对角线：1】
        #  for key in event chain dict:
        adj = np.zeros((self.n_nodes[split], self.n_nodes[split]))
        for key in dict:
            nodes = dict[key]
            mask = itertools.product(nodes, nodes)
            rows, cols = zip(*mask)
            adj[rows, cols] = 1
        return ((adj + adj.T)>0)[idx, :][:, idx]


def get_examples_indices(target_adj):
    # target_adj: indices x indices
    
    tri_target_adj = np.triu(target_adj, 1)

    true_indices = np.where(tri_target_adj>0)

    false_indices_all = np.where(tri_target_adj==0)
    mask = np.arange(0, len(false_indices_all[0]))
    np.random.shuffle(mask)
    false_indices = (false_indices_all[0][mask[:len(true_indices[0])]], false_indices_all[1][mask[:len(true_indices[0])]])

    assert len(true_indices[0]) == len(false_indices[0])
    return true_indices, false_indices

    # def ismember(a, b, tol=5):
    #     rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    #     return np.any(rows_close)

    # false_indices = []
    # while len(false_indices) < len(true_indices):
    #     idx_i = np.random.randint(0, target_adj.shape[0])
    #     idx_j = np.random.randint(0, target_adj.shape[0])
    #     if idx_i == idx_j:
    #         continue
    #     if idx_i > idx_j:
    #         idx_i, idx_j = idx_j, idx_i
    #     if ismember([idx_i, idx_j], true_indices):
    #         continue
    #     if false_indices:
    #         if ismember([idx_i, idx_j], np.array(false_indices)):
    #             continue
    #     false_indices.append((idx_i, idx_j))

    # false_indices_tup = zip(false_indices_ind[0], false_indices_ind[1])
    # false_indices_list = list(false_indices_tup)
    # print("####1", len(false_indices_list), false_indices_list[:10])
    # np.random.shuffle(false_indices_list)
    
    # return true_indices, zip(*false_indices_list)
    # return true_indices, false_indices
    