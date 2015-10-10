import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean

with open('pickled_data', 'rb') as read:
    all_data = pickle.load(read)
all_samples = all_data['matrix'].T
sample_ids = all_data['ids']

def PCA2D(X, ids):
    # Standardize
    X_std = StandardScaler().fit_transform(X.T)
    # PCA
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X_std)
    return combine(X_transf, ids)

def combine(X, ids, filename=None):
    all_data = []
    for i in range(len(X)):
        all_data += [[ids[i], X[i]]]
    if filename == None:
        return all_data
    else:
        f = open(filename, 'w')
        f.write(str(all_data))
        f.close()

def closest(X, point):
    smallest = [euclidean(point, X[0][1]), X[0]]
    distance = 0
    for points in X[1:]:
        distance = euclidean(point, points[1])
        if distance < smallest[0]: smallest = [distance, points[0]]
    return smallest
