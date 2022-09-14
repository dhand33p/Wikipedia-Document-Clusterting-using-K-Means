import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances

wiki = pd.read_csv('people_wiki.csv')

vectorizer = TfidfVectorizer(max_df=0.95)  # ignore words with very high doc frequency
tf_idf = vectorizer.fit_transform(wiki['text'])
words = vectorizer.get_feature_names()

tf_idf = csr_matrix(tf_idf)

tf_idf = normalize(tf_idf)


def get_initial_centroids(data, k, seed=None):
    """
    Randomly choose k data points as initial centroids
    """
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)

    n = data.shape[0]  # number of data points

    # Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)

    # Keep centroids as dense format, as many entries will be nonzero due to averaging.
    # As long as at least one document in a cluster contains a word,
    # it will carry a nonzero weight in the TF-IDF vector of the centroid.
    centroids = data[rand_indices, :].toarray()

    return centroids


distances = pairwise_distances(tf_idf, tf_idf[:3,:], metric='euclidean')
dist = distances[430,1]


closest_cluster = np.argmin(distances, axis=1)


def assign_clusters(data, centroids):
    """
    Parameters:
      - data      - is an np.array of float values of length n.
      - centroids - is an np.array of float values of length k.

    Returns
      -  A np.array of length n where the ith index represents which centroid
         data[i] was assigned to. The assignments range between the values 0, ..., k-1.
    """
    distances = pairwise_distances(data, centroids, metric='euclidean')
    return np.argmin(distances, axis=1)


def revise_centroids(data, k, cluster_assignment):
    """
    Parameters:
      - data               - is an np.array of float values of length N.
      - k                  - number of centroids
      - cluster_assignment - np.array of length N where the ith index represents which
                             centroid data[i] was assigned to. The assignments range between the values 0, ..., k-1.

    Returns
      -  A np.array of length k for the new centroids.
    """
    new_centroids = []
    for i in range(k):
        # Select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment == i]

        # Compute the mean of the data points.
        centroid = member_data_points.mean(axis=0)

        # Convert numpy.matrix type to numpy.ndarray type
        centroid = centroid.A1
        new_centroids.append(centroid)

    new_centroids = np.array(new_centroids)
    return new_centroids


def kmeans(data, k, initial_centroids, max_iter, record_heterogeneity=None, verbose=False):
    """
    This function runs k-means on given data and initial set of centroids.

    Parameters:
      - data                 - is an np.array of float values of length N.
      - k                    - number of centroids
      - initial_centroids    - is an np.array of float values of length k.
      - max_iter              - maximum number of iterations to run the algorithm
      - record_heterogeneity - if provided an empty list, it will compute the heterogeneity
                               at each iteration and append it to the list.
                               Defaults to None and won't record heterogeneity.
      - verbose              - set to True to display progress. Defaults to False and won't
                               display progress.

    Returns
      - centroids - A np.array of length k for the centroids upon termination of the algorithm.
      - cluster_assignment - A np.array of length N where the ith index represents which
                             centroid data[i] was assigned to. The assignments range between the
                             values 0, ..., k-1 upon termination of the algorithm.
    """
    centroids = initial_centroids[:]
    prev_cluster_assignment = None

    for itr in range(max_iter):
        # Print itereation number
        if verbose:
            print(itr)

        # 1. Make cluster assignments using nearest centroids
        cluster_assignment = assign_clusters(data, centroids)

        # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        centroids = revise_centroids(data, k, cluster_assignment)

        # Check for convergence: if none of the assignments changed, stop
        if prev_cluster_assignment is not None and \
                (prev_cluster_assignment == cluster_assignment).all():
            break

        # Print number of new assignments
        if prev_cluster_assignment is not None:
            num_changed = sum(abs(prev_cluster_assignment - cluster_assignment))
            if verbose:
                print(f'    {num_changed:5d} elements changed their cluster assignment.')

                # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data, k, centroids, cluster_assignment)
            record_heterogeneity.append(score)

        prev_cluster_assignment = cluster_assignment[:]

    return centroids, cluster_assignment


k = 3
q5_initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
q5_centroids, q5_cluster_assignment = kmeans(tf_idf, k, q5_initial_centroids, max_iter=400)


cluster_count = np.bincount(q5_cluster_assignment)
largest_cluster = np.argmax(cluster_count)


# Setup
def compute_heterogeneity(data, k, centroids, cluster_assignment):
    """
    Computes the heterogeneity metric of the data using the given centroids and cluster assignments.
    """
    heterogeneity = 0.0
    for i in range(k):

        # Select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment == i, :]

        if member_data_points.shape[0] > 0:  # check if i-th cluster is non-empty
            # Compute distances from centroid to data point
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances ** 2
            heterogeneity += np.sum(squared_distances)

    return heterogeneity


def smart_initialize(data, k, seed=None):
    """
    Use k-means++ to initialize a good set of centroids
    """
    if seed is not None:  # useful for obtaining consistent results
        np.random.seed(seed)

    centroids = np.zeros((k, data.shape[1]))

    # Randomly choose the first centroid.
    # Since we have no prior knowledge, choose uniformly at random
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx, :].toarray()

    # Compute distances from the first centroid chosen to all the other data points
    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()

    for i in range(1, k):
        # Choose the next centroid randomly, so that the probability for each data point to be chosen
        # is directly proportional to its squared distance from the nearest centroid.
        # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.
        idx = np.random.choice(data.shape[0], 1, p=distances / sum(distances))
        centroids[i] = data[idx, :].toarray()

        # Now compute distances from the centroids to all data points
        distances = np.min(pairwise_distances(data, centroids[0:i + 1], metric='euclidean'), axis=1)

    return centroids


def kmeans_multiple_runs(data, k, max_iter, seeds, verbose=False):
    """
    Runs kmeans multiple times

    Parameters:
      - data     - is an np.array of float values of length n.
      - k        - number of centroids
      - max_iter - maximum number of iterations to run the algorithm
      - seeds    - Either number of seeds to try (generated randomly) or a list of seed values
      - verbose  - set to True to display progress. Defaults to False and won't display progress.

    Returns
      - final_centroids          - A np.array of length k for the centroids upon
                                   termination of the algorithm.
      - final_cluster_assignment - A np.array of length n where the ith index represents which
                                   centroid data[i] was assigned to. The assignments range between
                                   the values 0, ..., k-1 upon termination of the algorithm.
    """
    min_heterogeneity_achieved = float('inf')
    final_centroids = None
    final_cluster_assignment = None
    if type(seeds) == int:
        seeds = np.random.randint(low=0, high=10000, size=seeds)

    num_runs = len(seeds)

    for seed in seeds:
        # Use k-means++ initialization:

        # Set record_heterogeneity=None because we will compute that once at the end.
        initial_centroids = smart_initialize(tf_idf, k, seed)

        # Run k-means:
        centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, max_iter=max_iter,
                                           record_heterogeneity=None, verbose=verbose)

        # To save time, compute heterogeneity only once in the end
        seed_heterogeneity = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)

        if verbose:
            print(f'seed={seed:06d}, heterogeneity={seed_heterogeneity:.5f}')

        # if current measurement of heterogeneity is lower than previously seen,
        # update the minimum record of heterogeneity.
        if seed_heterogeneity < min_heterogeneity_achieved:
            min_heterogeneity_achieved = seed_heterogeneity
            final_centroids = centroids
            final_cluster_assignment = cluster_assignment

    # Return the centroids and cluster assignments that minimize heterogeneity.
    return final_centroids, final_cluster_assignment


q7_centroids, q7_cluster_assignment = kmeans_multiple_runs(tf_idf, 5, max_iter=100, seeds=[20000, 40000, 80000])


q8_centroids, q8_cluster_assignment = kmeans_multiple_runs(tf_idf, 100, max_iter=400, seeds=[80000])
cluster_counts = np.bincount(q8_cluster_assignment)
num_small_clusters = sum(cluster_counts < 44)
