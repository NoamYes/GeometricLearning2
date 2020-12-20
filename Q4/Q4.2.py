import numpy as np
import powermethod


def verify_eigens(B):
    B_eigVals_truth, B_eigVecs_truth = np.linalg.eig(B)
    # Sort the eigenvalues and eigenvectors respectively (abs value)
    idx_pn = np.abs(B_eigVals_truth).argsort()[::-1]
    B_eigVals_truth = B_eigVals_truth[idx_pn]
    B_eigVecs_truth = B_eigVecs_truth[:, idx_pn]
    # Compute the eigenvalues/vectors from both PowerMethods
    u1, h1 = powermethod.PowerMethod(B, epsilon=1e-10)
    u2, h2 = powermethod.PowerMethod2(B, epsilon=1e-10)

    # Absolute of difference between 2 principal eigenvalues
    diff_h1 = abs(h1-B_eigVals_truth[0])
    diff_h2 = abs(h2-B_eigVals_truth[1])

    # Indicate distance between eigenvectors by norm - either +- v1
    diff_u1 = min(np.linalg.norm(B_eigVecs_truth[:, 0]-u1), np.linalg.norm(B_eigVecs_truth[:, 0] + u1))
    diff_u2 = min(np.linalg.norm(B_eigVecs_truth[:, 1]-u2), np.linalg.norm(B_eigVecs_truth[:, 1] + u2))

    return diff_h1, diff_h2, diff_u1, diff_u2


def examine_power_method(mat_num=100, mat_size=100, value_range=100):
    mat_num = int(np.ceil(mat_num))
    mat_size = int(np.ceil(mat_size))
    value_range = int(np.ceil(value_range))
    if mat_num <= 0 or mat_size <= 0 or value_range <= 0:
        raise Exception('valueError: Parameters must be positive')

    print('Testing convergence errors for h1, h2, u1, u2.')
    print('Errors are taken as the maximum out of ', mat_num, ' matrices of size [', mat_size, ', ', mat_size, ']')
    print('The matrices are uniformly distributed in range [', -value_range, ', ', value_range, ']')
    diffs = []
    for i in range(mat_num):
        B = np.random.uniform(-value_range, value_range, size=(mat_size, mat_size))
        B = (B + B.T)/2
        diff_h1, diff_h2, diff_u1, diff_u2 = verify_eigens(B)
        diffs.append([diff_h1, diff_h2, diff_u1, diff_u2])
    return np.max(diffs, axis=0)


max_diff = examine_power_method(1000, 100, 100)
print('h1 error <= ', '{0:.4g}'.format(max_diff[0]))
print('h2 error <= ', '{0:.4g}'.format(max_diff[1]))
print('u1 error <= ', '{0:.4g}'.format(max_diff[2]))
print('u2 error <= ', '{0:.4g}'.format(max_diff[3]))

print('end')
