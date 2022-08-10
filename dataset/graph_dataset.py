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
        self.event_idx = {'Train':[], 'Dev':[], 'Test':[]}  #所有节点(event+entity)中的inds
        self.entity_idx = {'Train':[], 'Dev':[], 'Test':[]}
        self.event_chain_dict = {'Train':{}, 'Dev':{}, 'Test':{}}  #{coref id(原数据):[objects]}
        self.entity_chain_dict = {'Train':{}, 'Dev':{}, 'Test':{}}  
        self.event_chain_list = {'Train':[], 'Dev':[], 'Test':[]}
        self.entity_chain_list = {'Train':[], 'Dev':[], 'Test':[]}  #按照节点编号的coref标签(index化了)list
        self.adjacency = {}  # 邻接矩阵, 节点:event mention(trigger), entity mention, 边:0./1.,对角线0,句子关系,文档关系,共指关系
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
            self.entity_idx[split] = list(set(range(self.n_nodes[split])) - set(self.event_idx[split]))

        #refine+后处理
        for split in ['Train']:
            #refine adj:加coref关系，对角线将为1
            self.refine_adj_by_event_coref(split)
            self.refine_adj_by_entity_coref(split)
            adj = self.adjacency[split] 
            self.adjacency[split] = np.where((adj+adj.T)>0, 1, 0)  

        #event, entity coref
        for split in ['Train', 'Dev', 'Test']:
            self.event_chain_list[split] = self.get_event_coref_list(split)
            self.entity_chain_list[split] = self.get_entity_coref_list(split)

        for split in ['Train']:
            self.event_coref_adj[split] = self.adjacency[split][self.event_idx[split], :][:, self.event_idx[split]]
            self.entity_coref_adj[split] = self.adjacency[split][self.entity_idx[split], :][:, self.entity_idx[split]]

        for split in ['Dev', 'Test']:
            self.event_coref_adj[split] = self.get_coref_adj(self.event_chain_dict[split], self.event_idx[split], split)  # bool矩阵, 对角线1
            self.entity_coref_adj[split] = self.get_coref_adj(self.entity_chain_dict[split], self.entity_idx[split], split)

        #对角线为0
        for split in ['Train', 'Dev', 'Test']:
            self.adjacency[split][np.diag_indices_from(self.adjacency[split])] = 0
        
        for split in ['Train', 'Dev', 'Test']:
            print("check", split)
            assert np.allclose(self.adjacency[split], self.adjacency[split], atol=1e-8)
            assert np.allclose(self.event_coref_adj[split],self.event_coref_adj[split].T,atol=1e-8)
            assert np.allclose(self.entity_coref_adj[split],self.entity_coref_adj[split].T,atol=1e-8)
        # print("check dataset########")
        # # 检查
        # for split in ['Train', 'Dev', 'Test']:
        #     print("#####",split)
        #     print("##adj")

        #     # print(self.adjacency[split].shape, self.adjacency[split])
        #     # print(self.event_coref_adj[split].shape)
        #     # print(self.entity_coref_adj[split].shape, self.entity_coref_adj[split])
            
        #     # print("##idx")
        #     # print(len(self.event_idx[split]))
        #     # print(len(self.entity_idx[split]))

        #     # print("##coref")
        #     # print(len(self.entity_chain_dict[split]), self.entity_chain_dict[split])
        #     # print(len(self.entity_chain_dict[split]), self.entity_chain_dict[split])
        #     # print(len(self.entity_chain_list[split]), self.entity_chain_list[split])
        #     # print(len(self.event_chain_list[split]), self.event_chain_list[split])

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
    

def get_examples_indices(target_adj):
    # target_adj: indices x indices
    #不取对角线
    
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
    
