import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def draw_nx_partition(path, descrip, G, partition, num=-1):
    plt.figure(figsize=(80,80), dpi=300)
    plt.title(descrip)
    dir = os.path.join(path, descrip) + str(num) + '.png'

    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                        cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    
    plt.savefig(dir)
    plt.close()
