from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

'''
    clustering is better when dbi closing to 0
'''

#Davies Bouldin Index
def eval_dbi(data, assignments):
    dbi = davies_bouldin_score(data, assignments)
    return dbi
    
def test():
    # Generating the sample data from make_blobs
    # This particular setting has one distinct cluster and 3 clusters placed close together.
    X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=4,
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1,
    )  # For reproducibility
    
    range_n_clusters = [4]

    for n_clusters in range_n_clusters:
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        print(len(cluster_labels))
        dbi = davies_bouldin_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The davies_bouldin_score is :",
            dbi,
        )
    
if __name__=='__main__':
    test()