import torch
from utils.eval import *
import community as community_louvain
import leidenalg as la
import igraph as ig


def louvain(G):
    #G: nx graph
    partition = community_louvain.best_partition(G)
    return partition


def leiden(G, max_size=-1):
    if max_size > 0:
        part = la.find_partition(G, la.ModularityVertexPartition, max_comm_size=max_size) # 限定最大子社区规模为10
    else:
        part = la.find_partition(G, la.ModularityVertexPartition)
    return part
