import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

with open('pickled_data', 'rb') as read:
    all_data = pickle.load(read)
all_samples = all_data['matrix'].T
sample_ids = all_data['ids']

def scikit_pca(X):
    # Standardize
    X_std = StandardScaler().fit_transform(X.T)
    # PCA
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X_std)
    return X_transf
    # Plot the data
    plt.scatter(X_transf[:,0], X_transf[:,1])
    plt.title('PCA via scikit-learn (using SVD)')
    plt.show()

def plot3d(all_samples, sample_ids=None, distort=False):
    mean_vector = np.mean(all_samples, axis=1)
    scatter_matrix = np.zeros((8,8))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:,i].reshape(8,1) - mean_vector).dot(
            (all_samples[:,i].reshape(8,1) - mean_vector).T)
    cov_mat = np.cov(all_samples)
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:,i].reshape(1,8).T
        eigvec_cov = eig_vec_cov[:,i].reshape(1,8).T
        assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'
    for i in range(len(eig_val_sc)):
        eigv = eig_vec_sc[:,i].reshape(1,8).T
        np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv),
                                             eig_val_sc[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
                 for i in range(len(eig_val_sc))]
    eig_pairs.sort()
    eig_pairs.reverse()
    matrix_w = np.hstack((eig_pairs[0][1].reshape(8,1),
                          eig_pairs[1][1].reshape(8,1),
                          eig_pairs[2][1].reshape(8,1)))
    transformed = matrix_w.T.dot(all_samples)
    if distort:
        minimum = np.amin(transformed, axis=1)
        transformed = transformed.T
        for i in range(len(transformed)):
            transformed[i] = transformed[i] - minimum
        maximum = np.amax(transformed.T, axis = 1)
        for i in range(len(transformed)):
            transformed[i] = transformed[i]*127/maximum
        transformed = transformed.T
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(transformed[0,:], transformed[1,:], transformed[2,:],
            'o', markersize=8, color='green', alpha=0.2)
    plt.show()

def plot2d(all_samples, sample_ids=None, distort=False):
    mean_vector = np.mean(all_samples, axis=1)
    scatter_matrix = np.zeros((8,8))
    for i in range(all_samples.shape[1]):
        scatter_matrix += (all_samples[:,i].reshape(8,1) - mean_vector).dot(
            (all_samples[:,i].reshape(8,1) - mean_vector).T)
    cov_mat = np.cov(all_samples)
    eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    for i in range(len(eig_val_sc)):
        eigvec_sc = eig_vec_sc[:,i].reshape(1,8).T
        eigvec_cov = eig_vec_cov[:,i].reshape(1,8).T
        assert eigvec_sc.all() == eigvec_cov.all(), 'Eigenvectors are not identical'
    for i in range(len(eig_val_sc)):
        eigv = eig_vec_sc[:,i].reshape(1,8).T
        np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv),
                                             eig_val_sc[i] * eigv,
                                             decimal=6, err_msg='', verbose=True)
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i])
                 for i in range(len(eig_val_sc))]
    eig_pairs.sort()
    eig_pairs.reverse()
    matrix_w = np.hstack((eig_pairs[0][1].reshape(8,1),
                          eig_pairs[1][1].reshape(8,1)))
    transformed = matrix_w.T.dot(all_samples)
    if distort:
        minimum = np.amin(transformed, axis=1)
        transformed = transformed.T
        for i in range(len(transformed)):
            transformed[i] = transformed[i] - minimum
        maximum = np.amax(transformed.T, axis = 1)
        for i in range(len(transformed)):
            transformed[i] = transformed[i]*127/maximum
        transformed = transformed.T
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.plot(transformed[0,:], transformed[1,:],
            'o', markersize=8, color='green', alpha=0.2)
    plt.show()
