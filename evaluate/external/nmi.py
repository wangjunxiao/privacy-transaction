import math
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

'''
    nmi in [0,1], clustering is better when closing to 1
'''

def NMI(A,B):
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #mutual information
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    #normalized mutual information
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

def eval_nmi(labels, assignments):
    result_NMI = normalized_mutual_info_score(labels, assignments)
    #result_NMI = NMI(labels, assignments)
    return result_NMI


if __name__ == '__main__':
    A = np.array([1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
    B = np.array([1,2,1,1,1,1,1,2,2,2,2,3,1,1,3,3,3])
    print(NMI(A,B))
    print(normalized_mutual_info_score(A,B))