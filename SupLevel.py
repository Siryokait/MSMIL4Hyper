import numpy as np
import copy
import dataAnalysis
from hyperDataSplit import hyper_Data_Seg, hyperdataSplit
from bagTranform import bagEmbed, miVLAD, miFAFV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV
from bagGraph import bagGraph, localbagGraph
from OVOClassifier import ovoBRSVM, ovoSVM, ovoBRSVMbs
import lcmr_functions as lcmr
import HMS
import time


def wrappedSup(hyper_data_d, supload, maindir, K, gt_img, gt_n, tr_ratio, C, gamma, length, width, n1, n2,
               method, seed, dataset):
    Whole_seg_imgs = []
    MS_HMS = []
    HMS_CN = []
    variance_required = 0.998
    spectral_data_pca = dataAnalysis.principal_component_extraction(hyper_data_d, variance_required)
    if supload == 1:
        HMS_Label = np.loadtxt(maindir + str(K) + ".txt", delimiter=",")
        MS_HMS.append(np.array(HMS_Label).astype(int))
        print(np.shape(HMS_Label))
        HMS_CN.append(int(np.max(HMS_Label.reshape(1, -1))) + 1)
        Whole_seg_imgs.append(np.array(HMS_Label).astype(int))
    else:
        spectral_mnf = lcmr.dimensional_reduction_mnf(hyper_data_d, 20)  # 20
        lcmr_matrices = lcmr.create_logm_matrices(spectral_mnf, 25, 400)  # 25 400

        hms = HMS.HMSProcessor(image=spectral_data_pca, lcmr_m=lcmr_matrices, k=K, m=4, a_1=0.5, a_2=0.5, mc=True)
        labels = hms.main_work_flow()
        Whole_seg_imgs.append(labels)
        MS_HMS.append(hms)
        print(hms.Labels)
        np.savetxt(maindir + str(K) + ".txt", hms.Labels, fmt="%f", delimiter=",")
        HMS_Label = hms.Labels

    spec4supmi = hyper_data_d
    tr_pixcos, te_pixcos, rest_pixcos = hyperdataSplit(gt_img, gt_n, tr_ratio, seed, dataset)

    # superpixel level ---------------------------------------
    gt_img4sup_extr = gt_img if dataset != 'UH' else gt_img[0]
    supdata, supxys, trsupidx, trsuplabel, tesupidx, errorsup = superpixelExtract(spec4supmi, HMS_Label, gt_img4sup_extr,
                                                                                  tr_pixcos)
    supins, supbag_ids, bgxys = Sup2Mi(supdata, supxys)
    supbgids = bagid_convert(supbag_ids)

    t1 = time.time()
    ESrep = bagEmbed(np.array(supins), np.array(supbgids), np.array(supbag_ids), N1=n1, N2=n2, type='hybrid')
    t2 = time.time()
    embed_t = t2 - t1
    print("Process time: ", t2 - t1)
    pca_es = PCA(n_components=100)
    # ESrep = pca_es.fit_transform(ESrep)

    trESrep = ESrep[trsupidx]
    trxys = []
    for tridx in trsupidx:
        trxys.append(supxys[tridx])

    teESrep = ESrep[tesupidx]
    texys = []
    for teidx in tesupidx:
        texys.append(supxys[teidx])

    if method == 'svm':
        # svm = GridSearchCV(SVC(probability=True), param_grid={"kernel": ("linear", 'rbf'), "C": np.logspace(-3, 3, 7),
        #                                       "gamma": np.logspace(-3, 3, 7)})
        svm = SVC(C=C, gamma=gamma, decision_function_shape='ovo', probability=True)
        svm.fit(trESrep, trsuplabel)
        t1 = time.time()
        predSuplabel = svm.predict(teESrep)
        supproba = svm.predict_proba(teESrep)
        t2 = time.time()
        inference_t = t2 - t1
        print("inference time", inference_t)

    supprobaidx = np.argsort(-supproba, axis=1)
    # print("superpixel proba.: ", supproba)
    shighest = supprobaidx[:, 0]
    ssecond = supprobaidx[:, 1]
    predsupcer = np.zeros(shape=np.shape(supproba)[0])
    print(np.max(supproba, axis=1))
    for i, h in enumerate(shighest):
        predsupcer[i] = (supproba[i, h] - supproba[i, ssecond[i]])
    # print("Superpixel cer. : ", predsupcer)

    res_map, supcermap = milab2segmap_sup(trsuplabel, trxys, predSuplabel, predsupcer, texys, length, width)
    return res_map, supcermap, tr_pixcos, te_pixcos, rest_pixcos, inference_t
