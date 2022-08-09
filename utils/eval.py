import numpy as np
import igraph as ig
import networkx as nx
from sklearn.metrics import label_ranking_loss, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi

from utils.train import sigmoid
from utils.visual import *
from models.clustring import leiden, louvain
from utils.plot import draw_nx_partition

import json
import pickle


def format_metrics(metrics, split):
    # f_score, roc_score, ap_score, p, r
    str = 'AUC={:.5f}, AP={:.5f}'.format(metrics[0], metrics[1])
    return str


def format_b3_metrics(metrics):
    # f_score, roc_score, ap_score, p, r
    str = 'R={:.5f}, P={:.5f}, F1={:.5f}'.format(metrics[0], metrics[1], metrics[2])
    return str


def test_model(emb, indices, true_indices, false_indices):
    # 计算: AUC, AP

    # 根据共指关系计算AUC等
    # 大矩阵: embedding, 共指关系矩阵
    # event mention在大矩阵中的下标,用于提取正负例,方法 取上三角矩阵(不含对角线)
    # extract event mentions(trigger)

    emb_ = emb[indices, :]
    # target_event_adj = target_adj[event_idx, :][:, event_idx]

    # Predict on test set of edges
    pred_adj = sigmoid(np.dot(emb_, emb_.T))

    # mask = np.triu_indices(len(indices), 1)  # 上三角元素的索引list
    # preds = pred_event_adj[mask]
    # target = target_sub_adj[mask]

    preds_true = pred_adj[true_indices]
    preds_false = pred_adj[false_indices]

    # np.random.shuffle(preds_false)
    # preds_false = preds_false[:len(preds_true)] # 正:负=1:1

    preds_all = np.hstack([preds_true, preds_false])
    labels_all = np.hstack([np.ones(len(preds_true)), np.zeros(len(preds_false))])

    # 计算metrics
    auc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    # f_score = get_bcubed(labels_all, preds_all>threshold)
    # p = precision_score(labels_all, preds_all>threshold)
    # r = recall_score(labels_all, preds_all>threshold) 
    return auc_score, ap_score


def eval_model_louvain(path, split, emb, indices=None, threshold=0.5, num=-1, visual=False):
    # embedding -> event cluster -> 可视化
    # return pred list(label list), n_comm

    # print('\tlouvain')
    emb_ = emb[indices, :]
    event_adj = sigmoid(np.dot(emb_, emb_.T))

    G, n_edges = adj_to_nx(event_adj, threshold)  # 01图
    # dir = os.path.join(path, 'g.pickle')
    # with open(dir, 'wb') as f:
    #     pickle.dump(G, f)

    partition = louvain(G)
    n_comm = max(partition.values()) + 1
    # print('\t\tcommunity=', n_comm)
    
    if visual:
        draw_nx_partition(path, split+' event clusters', G, partition, num)
    return list(partition.values()), n_comm, n_edges


def eval_model_leiden(path, split, emb, indices=None, threshold=0.5, num=-1, visual=False):
    # print('\tleiden')
    emb_ = emb[indices, :]
    event_adj = sigmoid(np.dot(emb_, emb_.T))
    G, n_edges = adj_to_ig(event_adj, threshold)  # 01图
    partition = leiden(G)
    n_comm = len(partition)

    # print('\t\tcommunity=', n_comm)
    if visual:
        dir = os.path.join(path, split+' event clusters') + str(num) + '.png'
        ig.plot(partition, dir)

    return partition.membership, n_comm, n_edges
    

def visual_graph(path, split, orig, pred_adj, num=-1, threshold=0.5):  # 输入邻接矩阵(原图, 预测图), 画出graph

    # plot_adj(path, split+" original visual graph", orig, num)  # 原图
    
    pred_adj_ = np.where(pred_adj>threshold, 1, 0)
    nuclear_norm = np.linalg.norm(pred_adj_, ord='nuc')
    print("\t\tnuclear norm/rank:", nuclear_norm)
    plot_adj(path, split+" pred graph - visual", pred_adj_, num=num)

    # plot_adj(path, split+" weighted pred visual graph", pred_adj, num=num, weighted=True)


def degree_analysis(path, split, orig, pred_adj, num=-1, threshold=0.5):

    degree = np.sum(orig, axis=1).astype(np.int)
    # degree_list_ = np.bincount(degree)
    max_degree = np.max(degree)
    min_degree = np.min(degree)
    mean_degree = np.mean(degree)
    median_degree = np.median(degree)
    print("\t\torig graph degree:", '\tmean:', mean_degree, '\tmedian:', median_degree, '\tmax:', max_degree, '\tmin', min_degree)

    # plot_hist(path, split+"original degree graph", degree_list_, num=num)

    adj = np.where(pred_adj>threshold, 1., 0.)
    pred_degree = np.sum(adj, axis=1).astype(np.int)
    # degree_list = np.bincount(pred_degree)  # 索引:度, 值:count

    max_degree = np.max(pred_degree)
    min_degree = np.min(pred_degree)
    mean_degree = np.mean(pred_degree)
    median_degree = np.median(pred_degree)
    print("\t\tpred graph degree:", '\tmean:', mean_degree, '\tmedian:', median_degree, '\tmax:', max_degree, '\tmin', min_degree)

    plot_hist(path, split+" original graph - degree", split+" pred graph - degree", degree, pred_degree, num=num)


def adj_to_nx(adj, threshold=0.5):

    G = nx.Graph()
    # adj = np.where(adj>threshold, 1., 0.)
    ind = np.where(np.triu(adj, 1)>threshold)
    # print("\t\t#edges=", len(ind[0]))
    # print("####", type(ind), ind)
    edges = zip(ind[0], ind[1])
    G.add_nodes_from(list(range(adj.shape[0])))
    G.add_edges_from(edges)
    return G, len(ind[0])


def adj_to_ig(adj, threshold=0.5):

    G = ig.Graph()
    # adj = np.where(adj>threshold, 1., 0.)
    ind = np.where(np.triu(adj, 1)>threshold)
    # print("\t\t#edges=", len(ind[0]))
    # print("####", type(ind), ind)
    edges = zip(ind[0], ind[1])
    # G = ig.Graph(edges)
    G.add_vertices(adj.shape[0])
    G.add_edges(edges)
    return G, len(ind[0])


def cal_nmi(labels_true, labels_pred):
    return nmi(labels_true, labels_pred)