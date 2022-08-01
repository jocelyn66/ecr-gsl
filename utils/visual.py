import matplotlib.pyplot as plt
import os

import numpy as np
from utils.evaluate import sigmoid
import networkx as nx


def plot(path, num, train_loss, dev_loss, test_loss):
    
    dir = os.path.join(path, 'converg') + str(num)

    fig, ax = plt.subplots(4,1,dpi=100, figsize=(12,18))
    epoch = list(range(len(train_loss)))
    ax[0].plot(epoch, train_loss, label='Train', color='#4472C4')
    ax[0].plot(epoch, dev_loss, label='Dev', color='#C00000')
    ax[0].plot(epoch, test_loss, label='Test', color='green')
    ax[0].legend(loc='best')

    ax[1].plot(epoch, train_loss, color='#4472C4')
    ax[1].set_title('Train')
    ax[2].plot(epoch, dev_loss, color='#C00000')
    ax[2].set_title('Dev')
    ax[3].plot(epoch, test_loss, color='green')
    ax[3].set_title('Test')
    # ax.set_title(name)
    # plt.grid(b=True, axis='y')

    # ax.grid(True)
    fig.tight_layout()

    # plt.imshow()
    plt.savefig(dir, bbox_inches='tight')

    plt.pause(10)
    plt.close(fig)
    
def plot1(path, num, train_loss):
    
    dir = os.path.join(path, 'converg') + str(num)

    fig, ax = plt.subplots(1,1,dpi=100, figsize=(12,18))
    epoch = list(range(len(train_loss)))
    ax[0].plot(epoch, train_loss, label='Train', color='#4472C4')
    ax[0].legend(loc='best')
    # ax.set_title(name)
    # plt.grid(b=True, axis='y')

    # ax.grid(True)
    fig.tight_layout()

    # plt.imshow()
    plt.savefig(dir, bbox_inches='tight')

    plt.pause(10)
    plt.close(fig)

def plot_adj(path, descrip, adj, num=-1, weighted=False):  # 画adj

    # networks

    G = nx.Graph()
    if not weighted:
        ind = np.where(np.triu(adj, 1)>0)
        print("#edges=", len(ind[0]))
        # print("####", type(ind), ind)
        edges = zip(ind[0], ind[1])
        G.add_edges_from(edges)

    # else (x, y, w)
    plt.figure(figsize=(80,80), dpi=300)
    plt.title(descrip)
    dir = os.path.join(path, descrip) + str(num) + '.png'
    pos = nx.spring_layout(G)

    if not weighted:
        # nx.draw(G, pos, node_color='b', edgelist=edges, width=1.0, edge_cmap=plt.cm.Blues, node_size=1)
        nx.draw(G, pos, node_color='b', width=1.0, edge_cmap=plt.cm.Blues, node_size=1)
    else:
        pass
        # nx.draw(G, pos, node_color='b', edgelist=edges, edge_color=weights, width=1.0, edge_cmap=plt.cm.Blues, node_size=1)
    # plt.savefig('./edges1.png')
    plt.savefig(dir)
    plt.close()


def plot_hist(path, descrip, descrip2, degree, degree2, num=-1):

    fig, ax = plt.subplots(2, 1, dpi=100, figsize=(40,30))

    ax[0].hist(degree, bins=200, color='b', alpha=0.3)
    ax[0].set_title(descrip)
    ax[0].set_xlabel('degree')
    # ax[0].set_ylabel('log-#nodes')

    ax[1].hist(degree2, bins=200, color='b', alpha=0.3)
    ax[1].set_title(descrip2)
    ax[1].set_xlabel('degree')
    # ax[1].set_ylabel('log-#nodes')

    # plt.yscale('log')
    fig.tight_layout()

    dir = os.path.join(path, descrip) + str(num) + '.png'
    plt.savefig(dir)
    plt.close()
