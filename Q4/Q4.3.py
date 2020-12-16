import numpy as np
import matplotlib.pyplot as plt 

# Init graph vertices for both path and ring
n = 30
D_n_size = 2*n
B_n_size = 3*n-1
D_n_nodes= np.arange(1,D_n_size)
B_n_nodes= np.arange(1,B_n_size) 

# Compute Adjacency matrices 
D_n_adj_mat = np.zeros((D_n_size,D_n_size))
B_n_adj_mat = np.zeros((B_n_size,B_n_size))

D_n_adj_mat[:n,:n] = 1
D_n_adj_mat[n-1,n] = 1
D_n_adj_mat[n,n-1] = 1
D_n_adj_mat[D_n_size-n:D_n_size, D_n_size-n:D_n_size] = 1
np.fill_diagonal(D_n_adj_mat, 0)

diag_idx = np.arange(B_n_size-1)
B_n_adj_mat[:n,:n] = 1
B_n_adj_mat[diag_idx,diag_idx+1] = 1
B_n_adj_mat[diag_idx+1,diag_idx] = 1
B_n_adj_mat[B_n_size-n:B_n_size,B_n_size-n:B_n_size] = 1
np.fill_diagonal(B_n_adj_mat, 0)

# Compute degree matrices 
D_n_deg_vec = np.sum(D_n_adj_mat, axis=0)
B_n_deg_vec = np.sum(B_n_adj_mat, axis=0)
D_n_deg_mat_sqrt = np.diag(D_n_deg_vec**(-1/2))
B_n_deg_mat_sqrt = np.diag(B_n_deg_vec**(-1/2))
D_n_deg_mat = np.diag(D_n_deg_vec)
B_n_deg_mat = np.diag(B_n_deg_vec)

# Compute Laplacian matrices 
D_n_lap_mat = D_n_deg_mat - D_n_adj_mat
B_n_lap_mat = B_n_deg_mat - B_n_adj_mat

# Compute Normalized Laplacian matrices 

D_n_normalized_lap_mat = D_n_deg_mat_sqrt.dot(D_n_lap_mat).dot(D_n_deg_mat_sqrt)
B_n_normalized_lap_mat = B_n_deg_mat_sqrt.dot(B_n_lap_mat).dot(B_n_deg_mat_sqrt)

# Compute lazy random walk matrices

D_n_WG_mat = 0.5*(np.eye(D_n_size)+D_n_adj_mat.dot(np.linalg.inv(D_n_deg_mat))) 
B_n_WG_mat = 0.5*(np.eye(B_n_size)+B_n_adj_mat.dot(np.linalg.inv(B_n_deg_mat))) 

# Compute eigen decomposition 

D_n_eigVals, D_n_eigVecs  = np.linalg.eig(D_n_normalized_lap_mat)
B_n_eigVals, B_n_eigVecs  = np.linalg.eig(B_n_normalized_lap_mat)

# Sort the eigen values with associated eigenvectors
D_n_sorted_NGL_idxs = D_n_eigVals.argsort()[::1] 
D_n_NGL_eigVals = D_n_eigVals[D_n_sorted_NGL_idxs] 
D_n_NGL_eigVecs = D_n_eigVecs[:,D_n_sorted_NGL_idxs]

B_n_sorted_NGL_idxs = B_n_eigVals.argsort()[::1] 
B_n_NGL_eigVals = B_n_eigVals[B_n_sorted_NGL_idxs] 
B_n_NGL_eigVecs = B_n_eigVecs[:,B_n_sorted_NGL_idxs]

# Plot the sorted eigenvalues
fig1 = plt.figure(1)
plt.xlabel('Statistical order')
plt.ylabel('Eigenvalue')
plt.title('Dumbbell graph, n=' +str(n) + ' sorted computed eigenvalues')

plt.plot(np.arange(len(D_n_NGL_eigVals))+1,D_n_NGL_eigVals)
# plt.show()

fig2 = plt.figure(2)
plt.xlabel('Statistical order')
plt.ylabel('Eigenvalue')
plt.title('Bolas graph, n=' +str(n) + ' sorted computed eigenvalues')

plt.plot(np.arange(len(B_n_NGL_eigVals))+1,B_n_NGL_eigVals)
# plt.show()

# Eigen decomposition of the lazy random walk matrix
D_n_WG_eigVals, D_n_WG_eigVecs  = np.linalg.eig(D_n_WG_mat)
B_n_WG_eigVals, B_n_WG_eigVecs  = np.linalg.eig(B_n_WG_mat)
D_n_sorted_idxs = D_n_WG_eigVals.argsort()[::-1] 
B_n_sorted_idxs = B_n_WG_eigVals.argsort()[::-1] 
D_n_WG_eigVals = D_n_WG_eigVals[D_n_sorted_idxs] 
D_n_WG_eigVecs = D_n_WG_eigVecs[:,D_n_sorted_idxs]
B_n_WG_eigVals = B_n_WG_eigVals[B_n_sorted_idxs] 
B_n_WG_eigVecs = B_n_WG_eigVecs[:,B_n_sorted_idxs]

# Plot the sorted eigenvalues
fig1 = plt.figure(3)
plt.xlabel('Statistical order')
plt.ylabel('Eigenvalue')
plt.title('Dumbbell graph Walk Matrix and Normalized graph laplcian eigenpairs, n=' +str(n))

plt.plot(np.arange(len(D_n_NGL_eigVals))+1,D_n_NGL_eigVals, label='Normalized graph laplcian')
plt.plot(np.arange(len(D_n_WG_eigVals))+1,D_n_WG_eigVals, label='Walk matrix')
plt.legend()
# plt.show()

fig2 = plt.figure(4)
plt.xlabel('Statistical order')
plt.ylabel('Eigenvalue')
plt.title('Bolas graph Walk Matrix and Normalized graph laplcian eigenpairs, n=' +str(n))

plt.plot(np.arange(len(B_n_NGL_eigVals))+1,B_n_NGL_eigVals, label='Normalized graph laplcian')
plt.plot(np.arange(len(B_n_WG_eigVals))+1,B_n_WG_eigVals, label='Walk matrix')
plt.legend()
# plt.show()

# Set initial mass distribution

rand_vertex_D_n = np.random.randint(0,D_n_size-1)
p0_D_n = np.zeros(D_n_size)
p0_D_n[rand_vertex_D_n] = 1
rand_vertex_B_n = np.random.randint(0,B_n_size-1)
p0_B_n = np.zeros(B_n_size)
p0_B_n[rand_vertex_B_n] = 1

# Run lazy walk function

def runWalkGraph(WG, p0, epsilon):
    step = 0
    breaking_condition = False
    p_list = []
    p_curr = p0
    while not breaking_condition:
        p_list.append(p_curr)
        p_new = np.dot(WG,p_curr)
        breaking_condition = np.linalg.norm(p_new-p_curr) < epsilon
        p_curr = p_new
        step = step + 1

    
    return p_list

# Apply walk function on both D_n and B_n 

epsilon = 1e-9
p_D_n_list = runWalkGraph(D_n_WG_mat, p0_D_n, epsilon)
p_B_n_list = runWalkGraph(B_n_WG_mat, p0_B_n, epsilon)

# Visualize distributions with time D_n

fig5, axs = plt.subplots(2,3, figsize=(15, 6))

axs = axs.ravel()
p_D_n_len = len(p_D_n_list)
p_B_n_len = len(p_B_n_list)
num_time_samples = 6 
time_samples = np.geomspace(1, p_D_n_len-1, num_time_samples, dtype=int)-1

for i, time in enumerate(time_samples):

    axs[i].scatter(np.arange(D_n_size), p_D_n_list[time], s=2)
    axs[i].set_title('Distribution at time=' + str(time))
    axs[i].set_ylim([-0.05,1.05])
    axs[i].set_xlabel('Vertex index')
    axs[i].set_ylabel('Probability')

fig5.suptitle('Lazy Walk probability distribution for D' +str(n) +' graph')

# Visualize distributions with time B_n

fig5, axs = plt.subplots(2,3, figsize=(15, 6))

axs = axs.ravel()
p_D_n_len = len(p_D_n_list)
p_B_n_len = len(p_B_n_list)
num_time_samples = 6 
time_samples = np.geomspace(1, p_B_n_len-1, num_time_samples, dtype=int)-1

for i, time in enumerate(time_samples):

    axs[i].scatter(np.arange(B_n_size), p_B_n_list[time], s=2)
    axs[i].set_title('Distribution at time=' + str(time))
    axs[i].set_ylim([-0.05,1.05])
    axs[i].set_xlabel('Vertex index')
    axs[i].set_ylabel('Probability')

fig5.suptitle('Lazy Walk probability distribution for B' +str(n) +' graph')

# plt.show()

# Demonstration of the convergence rate D_n

a = rand_vertex_D_n
b = [0, n, D_n_size-1, a]
stable_dist = p_D_n_list[-1]
d_a = D_n_deg_vec[a]
w2 = D_n_WG_eigVals[1]

time_vec = np.arange(len(p_D_n_list))
p_D_n_arr = np.array(p_D_n_list)

fig6, axs = plt.subplots(1,len(b), figsize=(15, 6))
axs = axs.ravel()

for i, b_vertex in enumerate(b):

    d_b = D_n_deg_vec[b_vertex]
    diff_from_stable = abs(p_D_n_arr[:,b_vertex]- stable_dist[b_vertex])
    bound_func = np.sqrt(d_b/d_a)*w2**time_vec
    axs[i].plot(time_vec, diff_from_stable, label=r'$|p_t(b)-\pi(b)|$')
    axs[i].plot(time_vec, bound_func, label= r'$\sqrt{\frac{d(b)}{d(a)}}w_2^t$')
    axs[i].set_yscale('log')
    axs[i].legend()
    if b_vertex == a:
        axs[i].set_title('Bound for vertex b = a')
    else:
        axs[i].set_title('Bound for vertex (b)=' +str(b_vertex))
    axs[i].set_ylim([-0.05,1.05])
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('logscale Difference')

fig6.suptitle('Upper bound convergence of stable probability, D' +str(n) +' graph')


# Demonstration of the convergence rate B_n

a = rand_vertex_B_n
b = [0, n, B_n_size-1, a]
stable_dist = p_B_n_list[-1]
d_a = B_n_deg_vec[a]
w2 = B_n_WG_eigVals[1]

time_vec = np.arange(len(p_B_n_list))
p_B_n_arr = np.array(p_B_n_list)

fig7, axs = plt.subplots(1,len(b), figsize=(15, 6))
axs = axs.ravel()

for i, b_vertex in enumerate(b):

    d_b = B_n_deg_vec[b_vertex]
    diff_from_stable = abs(p_B_n_arr[:,b_vertex]- stable_dist[b_vertex])
    bound_func = np.sqrt(d_b/d_a)*w2**time_vec
    axs[i].plot(time_vec, diff_from_stable, label=r'$|p_t(b)-\pi(b)|$')
    axs[i].plot(time_vec, bound_func, label= r'$\sqrt{\frac{d(b)}{d(a)}}w_2^t$')
    axs[i].set_yscale('log')
    axs[i].legend()
    if b_vertex == a:
        axs[i].set_title('Bound for vertex b = a')
    else:
        axs[i].set_title('Bound for vertex (b)=' +str(b_vertex))
    axs[i].set_ylim([-0.05,1.05])
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('logscale Difference')

fig7.suptitle('Upper bound convergence of stable probability, B' +str(n) +' graph')


plt.show()


print('ya')