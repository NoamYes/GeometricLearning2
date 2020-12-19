import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_3d_network(graph, angle, positions):
    pos = nx.get_node_attributes(graph, 'pos')

    with plt.style.context("bmh"):
        fig = plt.figure(figsize=(10, 7))
        ax = Axes3D(fig)
        for key, value in positions.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            ax.scatter(xi, yi, zi, edgecolor='b', alpha=0.9)
            for i, j in enumerate(graph.edges()):
                x = np.array((positions[j[0]][0], positions[j[1]][0]))
                y = np.array((positions[j[0]][1], positions[j[1]][1]))
                z = np.array((positions[j[0]][2], positions[j[1]][2]))
                ax.plot(x, y, z, c='black', alpha=0.9)
    ax.view_init(30, angle)
    plt.show()
