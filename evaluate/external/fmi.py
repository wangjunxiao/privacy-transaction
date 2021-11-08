import numpy as np
from sklearn.metrics import fowlkes_mallows_score

'''
    fmi in [0,1], the geometric mean of the Precision score and Recall score, 
    clustering is better when closing to 1
'''

#Fowlkes-Mallows Index
def eval_fmi(labels, assignments):
    result_FMI = fowlkes_mallows_score(labels, assignments)
    return result_FMI

def test():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])
    print(eval_fmi(y_true, y_pred))

if __name__ == '__main__':
    test()