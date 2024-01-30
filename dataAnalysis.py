import scipy.io as sio
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import time


def principal_component_extraction(spectral_original, variance_required):
    ## Variable List
    ## spectral_original: The original non reduced image
    ## variance_required: The required variance  ratio from 0 to 1

    ## Output list
    ## spectral_pc_final: The dimensional reduces image

    # 2d reshape
    spectral_2d = spectral_original.reshape(
        (spectral_original.shape[0] * spectral_original.shape[1], spectral_original.shape[2]))
    # Feature scaling preprocessing step
    spectral_2d = preprocessing.scale(spectral_2d)

    if (spectral_2d.shape[1] < 100):
        pca = PCA(n_components=spectral_2d.shape[1])
    else:
        pca = PCA(n_components=100)
    spectral_pc = pca.fit_transform(spectral_2d)
    explained_variance = pca.explained_variance_ratio_
    print(np.sum(explained_variance))

    if (np.sum(explained_variance) < variance_required):
        raise ValueError("The required variance was too high. Values should be between 0 and 1.")

    # Select the number of principal components that gives the variance required
    explained_variance_sum = np.zeros(explained_variance.shape)
    sum_ev = 0
    component_number = 0
    for i in range(explained_variance.shape[0]):
        sum_ev += explained_variance[i]
        if (sum_ev > variance_required and component_number == 0):
            component_number = i + 1
        explained_variance_sum[i] = sum_ev

    # Removed the unnecessary components and reshape in original 3d form
    spectral_pc = spectral_pc[:, :component_number]
    spectral_pc_final = spectral_pc.reshape((spectral_original.shape[0], spectral_original.shape[1], component_number))

    return spectral_pc_final