import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import scale
import scipy.linalg as lalg


def miVLAD(instances, bag_ids, K):
    kmeans = KMeans(n_clusters=K, random_state=1)
    kmeans.fit(instances)
    n_bags = np.max(bag_ids)
    n_instances = np.shape(instances)[0]
    n_featrues = np.shape(instances)[1]

    VLADs = []
    bag_id_now = bag_ids[0]
    temp = np.zeros(shape=(K, n_featrues))
    for i in range(n_instances):
        if bag_id_now != bag_ids[i]:
            ftemp = temp.flatten()
            pftemp = np.sign(ftemp) * np.sqrt(np.abs(ftemp))
            prftemp = pftemp / np.linalg.norm(pftemp)
            VLADs.append(prftemp)
            temp = np.zeros(shape=(K, n_featrues))
            bag_id_now = bag_ids[i]

        cluster_idx = kmeans.labels_[i]
        cluster_c = kmeans.cluster_centers_[cluster_idx]
        temp[cluster_idx] += instances[i] - cluster_c
    ftemp = temp.flatten()
    pftemp = np.sign(ftemp) * np.sqrt(np.abs(ftemp))
    prftemp = pftemp / np.linalg.norm(pftemp)

    VLADs.append(prftemp)
    VLADs = scale(VLADs, axis=1)
    return np.array(VLADs)


def miFAFV_sk(instances, bgids, N1):
    instances = scale(instances, axis=1)
    fa = FactorAnalysis(n_components=N1)
    zs = fa.fit_transform(instances)
    lambda_ = fa.components_.T
    psi = np.diag(fa.noise_variance_)
    miu_z_given_x = zs.T

    miu = np.mean(instances, axis=0)
    nf = np.shape(instances)[1]
    ni = np.shape(instances)[0]

    x = (instances - miu).T
    psi_inv = lalg.inv(psi)
    G_miu = (psi_inv @ (x - lambda_ @ miu_z_given_x)).T
    res = list()
    # explam = 0
    exptem = lalg.inv(psi + lambda_ @ lambda_.T)
    for idx in bgids:
        G_lambda = np.zeros(shape=(nf, N1))
        ni_in_bag = idx[1] - idx[0]
        for i in range(idx[0], idx[1]):
            xi = (instances[i] - miu).T.reshape(-1, 1)
            Ez = miu_z_given_x[:, i].reshape((-1, 1))
            G_lambda += ((psi_inv @ (xi - (lambda_ @ Ez))) @ Ez.T)

            # Ezz = np.identity(N1) - lambda_.T @ exptem @ lambda_ + Ez @ Ez.T
            # explam += psi_inv @ (xi @ Ez.T - lambda_ @ Ezz)
            # G_lambda += psi_inv @ (xi @ Ez.T - lambda_ @ Ezz)
        # t_F_lambda = lalg.svdvals(G_lambda) / ni_in_bag
        t_F_lambda = scale(G_lambda.reshape(-1) / ni_in_bag)
        t_F_miu = scale(np.sum(G_miu[idx[0]:idx[1]], axis=0) / ni_in_bag)
        t_FV = np.concatenate((t_F_miu, t_F_lambda), axis=0)
        res.append(t_FV)
    # res = scale(res, axis=1)
    # print("exp----", explam)
    return np.array(res)


def bagEmbed(instances, bgids, bag_id, N1, N2, type):
    if type == 'hybrid':
        ESres1 = miFAFV_sk(instances, bgids, N1)
        ESres2 = scale(miVLAD(instances=instances, bag_ids=bag_id, K=N2), axis=1)
        ESres = np.concatenate((ESres1, ESres2), axis=1)
        return ESres
    return -1
