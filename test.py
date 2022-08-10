"""Evaluation script."""

import argparse
import json
import os
import time

import torch
from tqdm import tqdm
import torch.nn.functional as F

import models
import numpy as np
from utils.name2object import name2model
from optimizers import *
from dataset.graph_dataset import GDataset

import torch.optim

import models.ecr_model
import optimizers.regularizers as regularizers
from optimizers.ecr_optimizer import *
from utils.train import *
from utils.visual import *
from dataset.dataset_process import preprocess_function
from dataset.graph_dataset import GDataset, get_examples_indices
from datasets import load_dataset
from utils.eval import *
from utils.bcubed_scorer import bcubed

import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    BertModel
)

DATA_PATH = './data/'

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--model-dir', '--dir', help="Model path")  # /home/h3c/00/running/logs/07_24/ecb+/gs_ecr-gsl_gae+_02_19_55/
parser.add_argument('--num', '--n', type=int, default=-1)
parser.add_argument('--threshold', type=float, default=0.5)


def test(model_dir, num=-1, threshold=0.5):
    
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(2022)

    args.use_cuda = torch.cuda.is_available()  # 有无
    # args.use_cuda = False
    if not args.use_cuda:
        ValueError("WARNING: CUDA is not available!!!")  # 用cpu的话, 直接注释
    args.device = torch.device("cuda" if args.use_cuda else "cpu")  # atth
    torch.cuda.manual_seed(args.seed)

    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    # t = time.time()

    dataset = GDataset(args)
    args.n_nodes = dataset.n_nodes

    pos_weight = {}
    norm = {}
    adj_norm = {}
    for split in ['Train', 'Dev', 'Test']:
        if not args.double_precision:
            dataset.adjacency[split] = dataset.adjacency[split].astype(np.float32)
        adj = dataset.adjacency[split]
        adj_norm[split] = preprocess_adjacency(adj)
        pos_w = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        if not args.double_precision:
            pos_w = pos_w.astype(np.float32)
        pos_weight[split] = torch.tensor(pos_w, device=args.device)
        norm[split] = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    
    # 问题: 重新sample neg, 每次算的metrics会有浮动
    event_true_sub_indices = {}
    event_false_sub_indices = {}
    entity_true_sub_indices = {}
    entity_false_sub_indices = {}
    recover_true_sub_indices = {}
    recover_false_sub_indices = {}

    for split in ['Train', 'Dev', 'Test']:
        event_true_sub_indices[split], event_false_sub_indices[split] = get_examples_indices(dataset.event_coref_adj[split])
        # entity_idx = list(set(range(args.n_nodes[split])) - set(dataset.event_idx[split]))
        entity_true_sub_indices[split], entity_false_sub_indices[split] = get_examples_indices(dataset.entity_coref_adj[split])
        recover_true_sub_indices[split], recover_false_sub_indices[split] = get_examples_indices(dataset.adjacency[split])

        # bert################################
     #Load Datasets
    data_files = {}
    data_files["train"] = args.train_file
    data_files["dev"] = args.dev_file
    data_files["test"] = args.test_file
    datasets = load_dataset("json", data_files=data_files)
    #Load Schema
    with open(args.schema_path, 'r') as f:
        schema_list = json.load(f)
        doc_schema = schema_list[0]
        event_schema = schema_list[1]
        entity_schema = schema_list[2]

    #introduce PLM
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    plm = BertModel.from_pretrained(args.plm_name)

    column_names = datasets["train"].column_names
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.train_cache_file
    )

    dev_dataset = datasets["dev"]
    dev_dataset = dev_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.dev_cache_file
    )

    test_dataset = datasets["test"]
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file= True,
        fn_kwargs={'tokenizer':tokenizer, 'args':args, 'schema_list':schema_list, 'plm':plm},
        cache_file_name = args.test_cache_file
    )

    datasets = {'Train': train_dataset, 'Dev':dev_dataset, 'Test':test_dataset}
    ######################

    # # load pretrained model weights
    # model = getattr(models, name2model[args.model])(args, tokenizer, plm, schema_list, dataset.adjacency['Train'])
    # if args.use_cuda:
    #     model.cuda()
    device = torch.device("cuda" if args.use_cuda else "cpu")

    model_name = "model{}_feat-d{}_h1-d{}_h2-d{}.pt".format(num, args.feat_dim, args.hidden1, args.hidden2)
    print(model_name)
    
    # model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    model = load_check_point(os.path.join(model_dir, model_name))
    model.to(device)

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = GAEOptimizer(args, model, optim_method, norm, pos_weight, args.use_cuda)

    model.eval()
    # eval#############
    # 计算边
    # 画图
    # 统计
    for split in ['Test', 'Train', 'Dev']:

        print(split)
        # print("读3个set,构造三张图:", time.time() - t)
        # t2 = time.time()
        _, mu = optimizer.eval(datasets[split], adj_norm[split], dataset.adjacency[split], split)
        # pred_adj = torch.sigmoid(mu@mu.T)
        # print("test预测:", time.time() - t2)
        hidden_emb = mu.data.detach().cpu().numpy()

        # cluster
        print("\tecr")
        pred_list2, n_comm2,n_edges = eval_model_leiden(model_dir, split, hidden_emb, dataset.event_idx[split], threshold, num)
        eval_metrics2 = bcubed(dataset.event_chain_list[split], pred_list2)
        print('threshold=', threshold, ', n_edges=', n_edges)
        print('\tleiden:', 'n_community=', n_comm2)
        print("\t\tb3 metrics:", format_b3_metrics(eval_metrics2))
        # print("\t\tnmi metric:", cal_nmi(dataset.event_chain_list[split], pred_list2))

        # pred_list, n_comm, n_edges = eval_model_louvain(model_dir, split, hidden_emb, dataset.event_idx[split], threshold, num)
        # eval_metrics = bcubed(dataset.event_chain_list[split], pred_list)
        # print("\t\tb3 metrics:", format_b3_metrics(eval_metrics))
        # print("\t\tnmi metric:", cal_nmi(dataset.event_chain_list[split], pred_list))

        # 计算边
        pred_adj = sigmoid(np.dot(hidden_emb, hidden_emb.T))

        # event coref#########
        print("\tevent coref:")
        orig_adj = dataset.event_coref_adj[split]
        pred_adj_ = pred_adj[dataset.event_idx[split], :][:, dataset.event_idx[split]]

        nuclear_norm = np.linalg.norm(pred_adj_, ord='nuc')
        print("\t\tnuclear norm/rank:", nuclear_norm)
        
        # degree_analysis(model_dir, split+' event coref', orig_adj, pred_adj_, num, threshold)
        # print("\tdegree analysis done")

        # # 可视化
        # visual_graph(model_dir, split+' event coref', orig_adj, pred_adj_, num, threshold)
        # print('\tvisual graph done')

        # entity coref##########
        print("\tentity coref:")
        orig_adj = dataset.entity_coref_adj[split]
        entity_idx = list(set(range(args.n_nodes[split])) - set(dataset.event_idx[split]))
        pred_adj_ = pred_adj[entity_idx, :][:, entity_idx]
        # degree_analysis(model_dir, split+' entity coref', orig_adj, pred_adj_, num, threshold)
        # print("\tdegree analysis done")

        # # 可视化
        # visual_graph(model_dir, split+' entity coref', orig_adj, pred_adj_, num, threshold)
        # print('\tvisual graph done')

        ##################################################
        # print('recover adj:')
        # # # metrics#########
        # # print('\tmetrics:')
        
        # # metrics1 = test_model(hidden_emb, dataset.event_idx[split], event_true_sub_indices[split], event_false_sub_indices[split])
        # # print("\t\tevent coref:" + format_metrics(metrics1, split))

        # # entity_idx = list(set(range(args.n_nodes[split])) - set(dataset.event_idx[split]))
        # # metrics2 = test_model(hidden_emb, entity_idx, entity_true_sub_indices[split], entity_false_sub_indices[split])
        # # print("\t\tentity coref:" + format_metrics(metrics2, split))

        # # metrics3 = test_model(hidden_emb, list(range(args.n_nodes[split])), recover_true_sub_indices[split], recover_false_sub_indices[split])
        # # print("\t\treconstruct adj:" + format_metrics(metrics3, split))

        # # 分析度
        # degree_analysis(model_dir, split+ ' recover', dataset.adjacency[split], pred_adj, num, threshold)
        # print("\tdegree analysis done")
        # # 可视化网络
        # visual_graph(model_dir, split+ ' recover', dataset.adjacency[split], pred_adj, num, threshold)
        # print('\tvisual graph done')


if __name__ == "__main__":
    print("test model")
    args = parser.parse_args()
    test(args.model_dir, args.num, args.threshold)
