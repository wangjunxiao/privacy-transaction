import random
import numpy as np
from dataloader import mnist
from kmeans_efficient_del import Kmeans, QKmeans, DCKmeans
from sklearn.metrics import silhouette_score
from kmeans_del import deletion

k=10
num_dels=20000

def evaluate(labels, data):
    kmeans = Kmeans(k, termination='fixed', iters=10)
    centers, assignments, loss = kmeans.run(data.copy())
    print(f'Clustering loss is {loss}')
    sc = silhouette_score(data, assignments)
    print(f'Silhouette coefficient is {sc}')

def main():
    labels, data = mnist.loader()
    evaluate(labels, data)
    #retrain from scratch and evaluate
    evaluate(deletion(num_dels, labels, data))
    
    
if __name__=='__main__':
    main()