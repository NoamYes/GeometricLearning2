import numpy as np
import powermethod


B = np.random.rand(10, 10)
B = (B + B.T)/2


def examine_power_method(matrices_num=100):
    diffs = []
    for i in range(matrices_num):
        B = np.random.rand(10, 10)
        B = (B + B.T)/2
        diff_h1, diff_h2, diff_u1, diff_u2 = verify_eigens(B)
        diffs.append([diff_h1, diff_h2, diff_u1, diff_u2])
    return np.max(diffs, axis=0)


def verify_eigens(B):
    B_eigVals_truth, B_eigVecs_truth = np.linalg.eig(B)
    # Sort the eigenvalues and eigenvectors respectively (abs value)
    idx_pn = np.abs(B_eigVals_truth).argsort()[::-1]
    B_eigVals_truth = B_eigVals_truth[idx_pn]
    B_eigVecs_truth = B_eigVecs_truth[:,idx_pn]
    # Compute the eigenvalues/vectors from both PowerMethods
    u1, h1 = powermethod.PowerMethod(B, epsilon=1e-10)
    u2, h2 = powermethod.PowerMethod2(B, epsilon=1e-10)

    # Absolute of difference between 2 principal eigenvalues
    diff_h1 = abs(h1-B_eigVals_truth[0])
    diff_h2 = abs(h2-B_eigVals_truth[1])

    # Indicate distance between eigenvectors by norm - either +- v1
    diff_u1 = min(np.linalg.norm(B_eigVecs_truth[:,0]-u1), np.linalg.norm(B_eigVecs_truth[:,0]+ u1))
    diff_u2 = min(np.linalg.norm(B_eigVecs_truth[:,1]-u2), np.linalg.norm(B_eigVecs_truth[:,1]+ u2))

    return diff_h1, diff_h2, diff_u1, diff_u2


diff_h1, diff_h2, diff_u1, diff_u2 = verify_eigens(B)
max_diff = test_power_method(matrices_num=100)

print('ya')
