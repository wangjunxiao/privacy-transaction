import numpy as np
from sklearn.metrics import v_measure_score

'''
    v in [0,1], is harmonic average between Homogeneity score and Completeness score, 
    clustering is better when closing to 1
'''

#V-Measure
def eval_v(labels, assignments):
    result_V = v_measure_score(labels, assignments)
    return result_V

def test():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 0, 1])
    print(eval_v(y_true, y_pred))

if __name__ == '__main__':
    test()