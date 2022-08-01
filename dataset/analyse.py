import numpy as np
import sys
import pickle as pkl
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import get_scheduler

DATA_PATH = './data/'

# np.random.seed(123)


class GDataset(object):

    def __init__(self, args, n_events=-1, n_entities=-1):

        self.name = args.dataset
        self.data = {}
        self.event_idx = {}
        self.entity_idx = {}
        self.node = {}
        self.event_chain_dict = {}
        # print(self.path)
        self.n_events = n_events
        self.n_entities = n_entities
        assert(self.n_events)
        assert(self.n_entities)
        self.n_sents = -1
        self.n_nodes = self.n_events + self.entities

        self.train_adjacency = self.get_adjacency(self, args.train_file)
        self.event_coref_adj = self.get_event_coref_adj()


    def get_adjacency(self, path, rand_rate=0.1):
        # 构图：
        # 节点：event, entity
        # 边：句子 文档关系
        # 【对角线：1】

        adj = np.zeros((self.n_nodes, self.n_nodes))
        last_doc_id = ''
        doc_node_idx = []
        sent_node_idx = []
        cur_idx = -1    # 从0开始顺序处理每个句子，对event chain, entity chain中的mention编号，根据mention出现的顺序
        
        with open(path, 'r') as f:
            lines = f.readlines() 

        for _, line in enumerate(lines):

            sent = json.loads(line)
            #  同一文档rand_rate的概率随机放点
            if last_doc_id != sent['sent_id']:
                rand_rows = doc_node_idx[np.random.rand(len(doc_node_idx))+1<-rand_rate*2]
                rand_cols = doc_node_idx[np.random.rand(len(doc_node_idx))+1<rand_rate*2]
                adj[rand_rows, rand_cols] = 1
                last_doc_id = sent['sent_id']
                doc_node_idx.clear()
            
            # event mentions
            for _, event in enumerate(sent['event_coref']):
                cur_idx += 1
                sent_node_idx.append(cur_idx)
                self.event_idx.append(cur_idx)
                self.add_new_item(self.event_chain_dict, event['coref_chain'], cur_idx)

            # eneity mentions
            sent_node_idx.append(range(len(sent['eneity_coref'])) + cur_idx)
            cur_idx += sent['eneity_coref']
            
            # 句子子图
            adj[sent_node_idx][sent_node_idx] = 1

            doc_node_idx.append(sent_node_idx)
            sent_node_idx.clear()

        # 处理adj
        adj = adj + adj.T
        assert(adj.diagonal(offset=0, axis1=0, axis2=1).all()>0)
        return adj>0      
        
    def get_event_node_idx(self, descrip):
        return int(self.schema_event[descrip])

    def get_entity_node_idx(self, descrip):
        return int(self.schema_entity[descrip]) + self.n_events

    def add_new_item(dict, item, idx):
        if not item in dict.keys():
            dict[item] = []
        dict[item].append(idx)

    def get_event_adjacency(self):
        # event coref关系0/1矩阵【对角线：1】
        #  for key in event chain dict:
        adj = np.zeros((self.n_nodes, self.n_nodes))
        for key in self.event_chain_dict:
            events = self.event_chain_dict[key]
            adj[events][events] = 1
        adj = adj + adj.T
        return adj > 0

    # def load_adjacency_sp_matrix(dataset, file, n_events, n_entities):
    #     n_nodes = n_events + n_entities
    #     # load train set
    #     # load train schema->event_idx, entity_idx
    #     for line:
    #         # 若doc结束:随机加边
    #         # 句子中的mention list:
    #         # 全排列
    #         # np.stack
        
    #     # npstack -> adj
    #     # 返回对称矩阵
    #     pass


def get_schema(path, split=''):
    # chain的schema, item：(chain descrip, id)
    if not split:
        ValueError
    with open(path, 'r') as x:
        schema = json.load(x)
    return schema[1], schema[2]


def check_coref_overlap(path, schema, split=''):
    # 构图：
    # 节点：event, entity
    # 边：句子 文档关系
    # 【对角线：1】
    
    with open(path, 'r') as f:
        lines = f.readlines() 

    for _, line in enumerate(lines):

        sent = json.loads(line)
        
        # event mentions
        for i, event in enumerate(sent['event_coref']):
            if event['coref_chain'] in schema:
                print('overlap in {}set'.format(split))
                print(i)
        print(_,'done')


def main():
    # 设置path
    train_schema_path = "../data/ECB+/processed_ECB+/ECB_Train_schema.json"
    dev_path = '../data/ECB+/processed_ECB+/ECB_Dev_processed_data.json'
    test_path = '../data/ECB+/processed_ECB+/ECB_Test_processed_data.json'

    # 读train schema
    train_schema = get_schema(train_schema_path)[0]
    # 读valid, test file
    # 检查
    check_coref_overlap(dev_path, train_schema, 'Dev')
    check_coref_overlap(test_path, train_schema, 'Test')


if __name__ == "__main__":
    main()
