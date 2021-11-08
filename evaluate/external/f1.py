import numpy as np
from sklearn.metrics import f1_score

'''
    f1 in [0,1], clustering is better when closing to 1
'''

#F1 Score
def eval_f1(labels, assignments, average='weighted'):
    result_F1 = f1_score(labels, assignments, average=average)
    return result_F1

def test():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])
    print(eval_f1(y_true, y_pred, average='macro'))
    print(eval_f1(y_true, y_pred, average='micro'))
    print(eval_f1(y_true, y_pred, average='weighted'))
    print(eval_f1(y_true, y_pred, average=None))

if __name__ == '__main__':
    test()