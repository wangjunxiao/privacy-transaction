import numpy as np
from sklearn.metrics import adjusted_rand_score

'''
    ari in [-1,1], clustering is better when closing to 1
'''

#Adjusted Rand Index
def eval_ari(labels, assignments):
    result_ARI = adjusted_rand_score(labels, assignments)
    return result_ARI

def test():
    labels_true = np.array([0, 0, 0, 1, 1, 1])
    labels_pred = np.array([0, 0, 1, 1, 2, 2])
    score = eval_ari(labels_true, labels_pred)
    print(score)

if __name__=='__main__':
    test()
