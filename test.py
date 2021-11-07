import numpy as np

n1 = 5*10**2
n2 = 5*10**2
n3 = 8*10**2

labels1 = np.array([0]*n1)
labels2 = np.array([1]*n2)
labels3 = np.array([2]*n3)

labels = np.concatenate((labels1, labels2, labels3))
print(len(labels))