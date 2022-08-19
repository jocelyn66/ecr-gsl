import itertools
import numpy as np
import json
import torch
import scipy.sparse as sp

from utils.train import add_new_item

DATA_PATH = './data/'

n_events = {'Train': 3808,'Dev': 1245, 'Test': 1780}
n_entities = {'Train': 4758,'Dev': 1476, 'Test': 2055}


class GDataset(object):

    def __init__(self, args):

        self.name = args.dataset
        self.event_idx = {'Train':[], 'Dev':[], 'Test':[]}  #所有节点(event+entity)中的inds
        self.entity_idx = {'Train':[], 'Dev':[], 'Test':[]}
        self.event_chain_dict = {'Train':{}, 'Dev':{}, 'Test':{}}  #{coref id(原数据):[objects]}
        self.entity_chain_dict = {'Train':{}, 'Dev':{}, 'Test':{}}  
        self.event_chain_list = {'Train':[], 'Dev':[], 'Test':[]}
        self.entity_chain_list = {'Train':[], 'Dev':[], 'Test':[]}  #按照节点编号的coref标签(index化了)list
        self.adjacency = {'Train':{}, 'Dev':{}, 'Test':{}}  # sp array, {'Train':{'event_coref', 'entity_coref', 'doc', 'sent'}, 'Train':{'doc', 'sent'}, 'Train':{'doc', 'sent'}}
        #邻接矩阵, 节点:event mention(trigger), entity mention
        #边(0./1.,对角线0):事件共指,实体共指,句子关系,实体关系
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
        
        #邻接矩阵:句子关系, 文档关系
        for split in ['Train', 'Dev', 'Test']:
            self.get_adj_sent_doc_rel(file[split], split)
            self.entity_idx[split] = list(set(range(self.n_nodes[split])) - set(self.event_idx[split]))

        #邻接矩阵:event共指,entity共指
        for split in ['Train']:
            self.get_adj_event_coref(split)
            self.get_adj_entity_coref(split)

        #event, entity coref list, for evaluation(b3...)
        for split in ['Train', 'Dev', 'Test']:
            self.event_chain_list[split] = self.get_event_coref_list(split)
            self.entity_chain_list[split] = self.get_entity_coref_list(split)

        # for split in ['Train']:
        #     self.event_coref_adj[split] = self.adjacency[split][self.event_idx[split], :][:, self.event_idx[split]]
        #     self.entity_coref_adj[split] = self.adjacency[split][self.entity_idx[split], :][:, self.entity_idx[split]]

        # for split in ['Dev', 'Test']:
        #     self.event_coref_adj[split] = self.get_coref_adj(self.event_chain_dict[split], self.event_idx[split], split)  # bool矩阵, 对角线1
        #     self.entity_coref_adj[split] = self.get_coref_adj(self.entity_chain_dict[split], self.entity_idx[split], split)

        # #对角线为0
        # for split in ['Train', 'Dev', 'Test']:
        #     self.adjacency[split][np.diag_indices_from(self.adjacency[split])] = 0
        
        #check对称
        for split in ['Train']:
            for s in ['event_coref', 'entity_coref', 'sent', 'doc']:
                assert check_sp_mx_symm(self.adjacency[split][s])

        for split in ['Dev', 'Test']:
            for s in ['sent', 'doc']:
                assert check_sp_mx_symm(self.adjacency[split][s])

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
        adj = np.where((adj + adj.T)>0, 1, 0)
        # adj[np.diag_indices_from(adj)] = 0
        return adj
        
    def get_event_node_idx(self, descrip):
        return int(self.schema_event[descrip])

    def get_entity_node_idx(self, descrip):
        return int(self.schema_entity[descrip]) + self.n_events[descrip]

    def get_coref_adj(self, dict, ind, split):
        # event coref关系bool矩阵【对角线：1】
        #  for key in event chain dict:
        adj = np.zeros((self.n_nodes[split], self.n_nodes[split]))
        for key in dict:
            events = dict[key]

            mask = itertools.product(events, events)
            rows, cols = zip(*mask)
            adj[rows, cols] = 1
        # adj = adj + adj.T   # 处理成对称矩阵

        return ((adj + adj.T)>0)[ind, :][:, ind]


    def get_event_coref_list(self, split):
        
        l = np.zeros(self.n_nodes[split])

        # chain的映射
        # chains = self.event_chain_dict[split].keys()
        # chains_set = set(chains)

        # chains_mapping = {}
        # for label in chains_set:
        #     chains_mapping[label] = len(chains_mapping)

        for i, chain in enumerate(self.event_chain_dict[split]):
            l[self.event_chain_dict[split][chain]] = int(i)
        #enumerate dict, list[i]=chain idx
        return l[self.event_idx[split]].astype(int).tolist()

    def get_entity_coref_list(self, split):

        l = np.zeros(self.n_nodes[split])
        for i, chain in enumerate(self.entity_chain_dict[split]):
            l[self.entity_chain_dict[split][chain]] = int(i)
        return l[self.entity_idx[split]].astype(int).tolist()

    def refine_adj_by_event_coref(self, split):
        #将(event, event)设置为0
        #遍历coref dict加边

        mask = itertools.product(self.event_idx[split], self.event_idx[split])
        rows, cols = zip(*mask)
        self.adjacency[split][rows, cols] = 0

        for key in self.event_chain_dict[split]:
            events = self.event_chain_dict[split][key]
            mask = itertools.product(events, events)
            rows, cols = zip(*mask)
            self.adjacency[split][rows, cols] = 1

    def refine_adj_by_entity_coref(self, split):
        
        mask = itertools.product(self.entity_idx[split], self.entity_idx[split])
        rows, cols = zip(*mask)
        self.adjacency[split][rows, cols] = 0

        for key in self.entity_chain_dict[split]:
            entitis = self.entity_chain_dict[split][key]

            mask = itertools.product(entitis, entitis)
            rows, cols = zip(*mask)
            self.adjacency[split][rows, cols] = 1
        
    def get_adj_event_coref(self, split):
        edges = []
        for key in self.event_chain_dict[split]:
            events = self.event_chain_dict[split][key]
            mask = itertools.product(events, events)
            edges.extend(list(mask))

        row, col = zip(*edges)
        values = np.ones(len(edges))  #type?
        mx = sp.coo_matrix((values, (row, col)), shape=(self.n_nodes[split], self.n_nodes[split]))
        self.adjacency[split]['event_coref'] = refine_adj(mx)

    def get_adj_entity_coref(self, split):
        edges = []
        for key in self.entity_chain_dict[split]:
            events = self.entity_chain_dict[split][key]
            mask = itertools.product(events, events)
            edges.extend(list(mask))

        row, col = zip(*edges)
        values = np.ones(len(edges))  #type?
        mx = sp.coo_matrix((values, (row, col)), shape=(self.n_nodes[split], self.n_nodes[split]))
        self.adjacency[split]['entity_coref'] = refine_adj(mx)

    def get_adj_sent_doc_rel(self, path, split):

        last_doc_id = ''
        doc_node_idx = []
        sent_node_idx = []
        doc_row = []
        doc_col = []
        sent_row = []
        sent_col = []
        cur_idx = -1    # 从0开始顺序处理每个句子，对event chain, entity chain中的mention编号，根据mention出现的顺序
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        for _, line in enumerate(lines):
            sent = json.loads(line)

            #文档关系
            if last_doc_id != sent['doc_id']:
                if len(doc_node_idx)>0:
                    indices = itertools.product(doc_node_idx, doc_node_idx)
                    row, col = zip(*indices)
                    doc_row.extend(list(row))
                    doc_col.extend(list(col))

                last_doc_id = sent['doc_id']
                doc_node_idx = []
            
            # event coref dict
            for _, event in enumerate(sent['event_coref']):
                cur_idx += 1
                sent_node_idx.append(cur_idx)
                self.event_idx[split].append(cur_idx)
                add_new_item(self.event_chain_dict[split], event['coref_chain'], cur_idx)

            # eneity coref dict
            for _, entity in enumerate(sent['entity_coref']):
                cur_idx += 1
                sent_node_idx.append(cur_idx)
                # self.entity_idx[split].append(cur_idx)
                add_new_item(self.entity_chain_dict[split], entity['coref_chain'], cur_idx)
            
            # 句子关系
            indices = itertools.product(sent_node_idx, sent_node_idx)
            row, col = zip(*indices)
            sent_row.extend(row)
            sent_col.extend(col)
            doc_node_idx.extend(sent_node_idx)
            sent_node_idx = []

        sent_mx = sp.coo_matrix((np.ones(len(sent_row)), (np.array(sent_row), np.array(sent_col))), shape=(self.n_nodes[split], self.n_nodes[split]))
        doc_mx = sp.coo_matrix((np.ones(len(doc_row)), (doc_row, doc_col)), shape=(self.n_nodes[split], self.n_nodes[split]))

        self.adjacency[split]['sent'] = refine_adj(sent_mx)
        self.adjacency[split]['doc'] = refine_adj(doc_mx)


def refine_adj(sp_mx):
    #对称,对角线置为0
    # sp_mx = sp_mx + sp_mx.T
    return (sp_mx - sp.eye(sp_mx.shape[0])).tocoo()


def get_examples_indices(sp_mx, idx):
    # func:for eval lp task, auc ap
    #不取对角线
    #sp_mx: sp mx or ndarray
    #return: ([],[])
    if sp.issparse(sp_mx):
        adj = sp_mx.toarray()
    else:
        adj = sp_mx
    adj = adj[idx, :][:, idx]
    adj = np.triu(adj, 1)
    true_edges = np.where(adj>0)

    false_adj = np.tri(adj.shape[0], adj.shape[0], -1).T
    false_adj = false_adj - adj
    false_indices_all = np.where(false_adj>0)
    mask = np.arange(0, len(false_indices_all[0]))
    np.random.shuffle(mask)
    false_edges = (false_indices_all[0][mask[:len(true_edges[0])]], false_indices_all[1][mask[:len(true_edges[0])]])

    assert len(true_edges[0]) == len(false_edges[0])
    return true_edges, false_edges


def check_sp_mx_symm(mx):
    adj = mx.toarray()
    return np.allclose(adj, adj.T, atol=1e-8)
