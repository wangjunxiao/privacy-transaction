import numpy as np

n1 = 1*10**2
n2 = 2*10**2
n3 = 2*10**2
n4 = 1*10**2
n5 = 2*10**2
n = n1 + n2 + n3 + n4 + n5
cov1 = [[0.3,0],[0,0.3]]
cov2 = [[0.3,0],[0,0.3]]
cov3 = [[0.3,0],[0,0.3]]
cov4 = [[0.3,0],[0,0.3]]
cov5 = [[0.4,0],[0,0.5]]

def loader():
    # synthetic dataset made from a Gaussian mixture
    center1 = np.array([1.5,0])
    center2 = np.array([0,1.5])
    center3 = np.array([-1.5,-1.5])
    center4 = np.array([0.6,-1.0])
    center5 = np.array([-1.5,1.5])
    cluster1 = np.random.multivariate_normal(center1, cov1, n1)
    cluster2 = np.random.multivariate_normal(center2, cov2, n2)
    cluster3 = np.random.multivariate_normal(center3, cov3, n3)
    cluster4 = np.random.multivariate_normal(center4, cov4, n4)
    cluster5 = np.random.multivariate_normal(center5, cov5, n5)
    labels1 = np.array([0]*n1)
    labels2 = np.array([1]*n2)
    labels3 = np.array([2]*n3)
    labels4 = np.array([3]*n4)
    labels5 = np.array([4]*n5)
    #centers = np.vstack((center1, center2, center3, center4, center5))
    data = np.vstack((cluster1, cluster2, cluster3, cluster4, cluster5))
    labels = np.concatenate((labels1, labels2, labels3, labels4, labels5))
    return labels, data