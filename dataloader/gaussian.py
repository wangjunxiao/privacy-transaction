import numpy as np

n1 = 2*10**2
n2 = 4*10**2
n3 = 8*10**2
n = n1 + n2 + n3
cov = [[0.05,0],[0,0.05]]

def loader():
    # synthetic dataset made from a Gaussian mixture
    cluster1 = np.random.multivariate_normal([2,0], cov, n1)
    cluster2 = np.random.multivariate_normal([0,2], cov, n2)
    cluster3 = np.random.multivariate_normal([-1.5,-1.5], cov, n3)
    labels1 = np.array([0]*n1)
    labels2 = np.array([1]*n2)
    labels3 = np.array([2]*n3)
    data = np.vstack((cluster1, cluster2, cluster3))
    labels = np.concatenate((labels1, labels2, labels3))
    return labels, data