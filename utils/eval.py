import numpy as np
import igraph as ig
import networkx as nx
from sklearn.metrics import label_ranking_loss, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

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


# cluster####################
def mult_precision(el1, el2, cdict, ldict):
    """Computes the multiplicity precision for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
        / float(len(cdict[el1] & cdict[el2]))


def mult_recall(el1, el2, cdict, ldict):
    """Computes the multiplicity recall for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
        / float(len(ldict[el1] & ldict[el2]))
        

def precision(cdict, ldict):
    """Computes overall extended BCubed precision for the C and L dicts."""
    return np.mean([np.mean([mult_precision(el1, el2, cdict, ldict) \
        for el2 in cdict if cdict[el1] & cdict[el2]]) for el1 in cdict])


def recall(cdict, ldict):
    """Computes overall extended BCubed recall for the C and L dicts."""
    return np.mean([np.mean([mult_recall(el1, el2, cdict, ldict) \
        for el2 in cdict if ldict[el1] & ldict[el2]]) for el1 in cdict])


def fscore(p_val, r_val, beta=1.0):
    """Computes the F_{beta}-score of given precision and recall values."""
    return (1.0 + beta**2) * (p_val * r_val / (beta**2 * p_val + r_val))   


def bcubed(gold_lst, predicted_lst):
    # in: gold_list: cluster set
    """
    Takes gold, predicted.
    Returns recall, precision, f1score
    """
    gold = {i:{cluster} for i,cluster in enumerate(gold_lst)}
    pred = {i:{cluster} for i,cluster in enumerate(predicted_lst)}
    p = precision(pred, gold)
    r = recall(pred, gold)
    return r, p, fscore(p, r)
    
###################################


def test_model(emb, indices, true_indices, false_indices):
    # target_adj: 

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


def eval_model_louvain(path, split, emb, indices=None, threshold=0.5, num=-1):
    # embedding -> event cluster -> 可视化 + 测评
    print('louvain')
    emb_ = emb[indices, :]
    event_adj = sigmoid(np.dot(emb_, emb_.T))

    G = adj_to_nx(event_adj, threshold)  # 01图

    # dir = os.path.join(path, 'g.pickle')
    # with open(dir, 'wb') as f:
    #     pickle.dump(G, f)

    partition = louvain(G)
    print('community=', max(partition.values()) + 1)

    draw_nx_partition(path, split+' event clusters', G, partition, num)
    # 测评??
    # partition -> label


def eval_model_leiden(path, split, emb, indices=None, threshold=0.5, num=-1):
    print('leiden')
    emb_ = emb[indices, :]
    event_adj = sigmoid(np.dot(emb_, emb_.T))
    G = adj_to_ig(event_adj, threshold)  # 01图
    partition = leiden(G)
    print('community=', len(partition))

    dir = os.path.join(path, split+' event clusters') + str(num) + '.png'
    ig.plot(partition, dir)
    

def visual_graph(path, split, orig, pred_adj, num=-1, threshold=0.5):  # 输入邻接矩阵(原图, 预测图), 画出graph

    # plot_adj(path, split+" original visual graph", orig, num)  # 原图
    
    pred_adj_ = np.where(pred_adj>threshold, 1, 0)
    nuclear_norm = np.linalg.norm(pred_adj_, ord='nuc')
    print("\tnuclear norm/rank:", nuclear_norm)
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
    print("#edges=", len(ind[0]))
    # print("####", type(ind), ind)
    edges = zip(ind[0], ind[1])
    G.add_edges_from(edges)
    return G


def adj_to_ig(adj, threshold=0.5):

    # adj = np.where(adj>threshold, 1., 0.)
    ind = np.where(np.triu(adj, 1)>threshold)
    print("#edges=", len(ind[0]))
    # print("####", type(ind), ind)
    edges = zip(ind[0], ind[1])
    G = ig.Graph(edges)
    return G
