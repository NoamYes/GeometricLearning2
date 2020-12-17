import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph

def AffinityMat(Z, kernel_method, epsilon, n_neighbor=None, **kwargs):
    N = Z.shape[0]   
    m = Z.shape[1]
    affinity_mat = np.zeros((N,N))

    if kernel_method == 'Gaussian' and n_neighbor == None:
        # variance = (1/(2*np.pi*epsilon)**(m/2))
        variance = 1
        kernel_func = lambda x, y: variance*np.exp(-(1/epsilon)*cdist(x,y,'sqeuclidean')) # Gaussian kernel    
    elif kernel_method == 'Gaussian' and not n_neighbor == None:
        # variance = (1/(2*np.pi*epsilon)**(m/2))
        variance = 1
        kernel_func = lambda x, y: variance*np.exp(-(1/epsilon)*np.linalg.norm(x-y)) # Gaussian kernel
    elif kernel_method == 'Linear':
        kernel_func = lambda x, y: np.dot(x,y.T)  # Linear kernel
    elif kernel_method == 'Polynomial':
        a = kwargs['a']
        b = kwargs['b']
        c = kwargs['c']
        kernel_func = lambda x, y: (np.dot(x,y.T)*a+b)**c  # Polynomial kernel
    else:
        raise ValueError("Invalid Kernel Type")

    if n_neighbor == None:
        affinity_mat = kernel_func(Z,Z)
    else:
        euc_dist_mat = cdist(Z,Z,'sqeuclidean')
        adj_mat = kneighbors_graph(euc_dist_mat, n_neighbors=n_neighbor)
        # Make the adj_mat symetric 
        adj_mat = np.maximum(adj_mat.toarray(), adj_mat.toarray().T)
        # Calculate kernel function on neighbors only
        res = np.where(adj_mat == 1)
        for pair in zip(res[0], res[1]):
            r, c = pair
            affinity_mat[r,c] = kernel_func(Z[r], Z[c])







    return affinity_mat
