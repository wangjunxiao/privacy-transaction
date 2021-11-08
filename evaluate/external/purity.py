import collections
import numpy as np

'''
    purity in [0,1], clustering is better when closing to 1
'''

def purity(result, label):
    total_num = len(label)
    cluster_counter = collections.Counter(result)
    original_counter = collections.Counter(label)
    t = []
    for k in cluster_counter:
        p_k = []
        for j in original_counter:
            count = 0
            for i in range(len(result)):
                if result[i] == k and label[i] == j: # joint set
                    count += 1
            p_k.append(count)
        temp_t = max(p_k)
        t.append(temp_t)
    
    return sum(t)/total_num

def eval_purity(labels, assignments):
    result_Purity = purity(labels, assignments)
    return result_Purity

if __name__ == '__main__':
    label  = np.array([0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 1, 1, 0, 2, 2, 2, 0])
    result = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    print(purity(label,result))