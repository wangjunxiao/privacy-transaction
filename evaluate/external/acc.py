import numpy as np
from sklearn.metrics import accuracy_score

'''
    acc in [0,1], 
    clustering is better when closing to 1
'''

#Cluster Accuracy
def eval_acc(labels, assignments):
    result_ACC = accuracy_score(labels, assignments)
    return result_ACC

def test():
    A = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
    B = np.array([1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 1, 1, 3, 3, 3])
    C = np.array([2, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 3, 2, 2, 3, 3, 3])  
    D = np.array([1, 3, 1, 1, 1, 1, 1, 3, 3, 3, 3, 2, 1, 1, 2, 2, 2])
    print(eval_acc(A, B)) 
    print(eval_acc(A, C)) 
    print(eval_acc(B, D)) 
    print(eval_acc(C, D)) 

if __name__ == '__main__':
    test()