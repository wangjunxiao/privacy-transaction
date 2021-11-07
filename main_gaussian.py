import random
import numpy as np
from dataloader import gaussian
from kmeans_efficient_del import Kmeans, QKmeans, DCKmeans
from sklearn.metrics import silhouette_score
from kmeans_del import deletion

k=3
num_dels=1000

def evaluate(labels, data):
    kmeans = Kmeans(k, termination='fixed', iters=10)
    centers, assignments, loss = kmeans.run(data.copy())
    print(f'Clustering loss is {loss}')
    sc = silhouette_score(data, assignments)
    print(f'Silhouette coefficient is {sc}')

def main():
    labels, data = gaussian.loader()
    print(f'processing request to delete {num_dels} from total {len(data)} points')
    evaluate(labels, data)
    #retrain from scratch and evaluate
    labels, data = deletion(num_dels, labels, data)
    evaluate(labels, data)
    

if __name__=='__main__':
    main()