import numpy as np
import copy
import dataAnalysis
from hyperDataSplit import hyperdataSplit
from bagTranform import bagEmbed
from sklearn.svm import SVC
import lcmr_functions as lcmr
import HMS
import time


def errorprocess(supxy, supidx, gtmap):
    gt_n = np.max(gtmap)
    count = np.zeros(shape=gt_n + 1)
    for xy in supxy[supidx]:
        x = xy[0]
        y = xy[1]
        count[gtmap[x, y]] += 1
    res = np.argmax(count[1:])
    # print("&&&&&", res + 1)
    return res + 1


def superpixelExtract(Hsi, Supmap, gtmap, trxys):
    Supmap = Supmap.astype(int)
    n_sup = np.max(np.reshape(Supmap, newshape=(-1, 1)), axis=0)
    Supxys = [[] for i in range(int(n_sup) + 1)]
    Supdata = [[] for i in range(int(n_sup) + 1)]
    trsupmark = np.zeros(shape=n_sup + 1, dtype=int)
    rows, cols = np.shape(Supmap)

    trgtmap = np.zeros(shape=(rows, cols), dtype=int)
    for xys in trxys:
        trgtmap[xys[0], xys[1]] = gtmap[xys[0], xys[1]]

    for i in range(rows):
        for j in range(cols):
            Supxys[Supmap[i, j]].append([i, j])
            Supdata[Supmap[i, j]].append(Hsi[i, j])
            if trgtmap[i, j] > 0 and trsupmark[Supmap[i, j]] == 0:
                trsupmark[Supmap[i, j]] = trgtmap[i, j]
            elif 0 < trgtmap[i, j] != trsupmark[Supmap[i, j]] and trsupmark[Supmap[i, j]] > 0:
                trsupmark[Supmap[i, j]] = -1
    trsupidx = []
    tesupidx = []
    trsuplabel = []
    errorsup = []
    for i, mark in enumerate(trsupmark):
        if mark > 0:
            trsupidx.append(i)
            trsuplabel.append(mark)
        else:
            tesupidx.append(i)
            # if mark == -1:
            #     trsupidx.append(i)
            #     trsuplabel.append(errorprocess(Supxys, i, gtmap))
            #     errorsup.append(i)
    return Supdata, Supxys, trsupidx, trsuplabel, tesupidx, errorsup


def Sup2Mi(Supdata, Supxys):
    bag_ids = []
    instances = []
    bgxys = []
    for i, bag in enumerate(Supdata):
        x = []
        y = []
        for j, instance in enumerate(bag):
            instances.append(instance)
            bag_ids.append(i)
            x.append(Supxys[i][j][0])
            y.append(Supxys[i][j][1])
        bgxys.append([np.mean(x), np.mean(y)])

    return instances, bag_ids, bgxys


def bagid_convert(bag_ids):
    n_ins = np.shape(bag_ids)[0]
    bgids = []
    this_idx = [0, 0]
    this_idx[0] = 0
    this_bgid = bag_ids[0]
    for i in range(n_ins):
        if bag_ids[i] != this_bgid:
            this_idx[1] = i
            bgids.append(copy.deepcopy(this_idx))
            this_idx[0] = i
            this_bgid = bag_ids[i]
    this_idx[1] = n_ins
    bgids.append(this_idx)
    return np.array(bgids)


def milab2segmap_sup(trlab, trxys, telab, tecer, texys, rows, cols):
    segmap = np.zeros(shape=(rows, cols), dtype=int)
    cermap = np.zeros(shape=(rows, cols))
    for i, xys in enumerate(trxys):
        for xy in xys:
            segmap[xy[0], xy[1]] = trlab[i]
            cermap[xy[0], xy[1]] = 1000

    for i, xys in enumerate(texys):
        for xy in xys:
            segmap[xy[0], xy[1]] = telab[i]
            cermap[xy[0], xy[1]] = tecer[i]

    return segmap, cermap


def segmap2acc(segmap, gtmap, tepixco):
    n_t = np.shape(tepixco)[0]
    n_gt = np.max(gtmap).astype(int)
    resmat = np.zeros(shape=(n_gt, n_gt), dtype=int)
    corr = 0
    catcorr = [0 for i in range(n_gt)]
    cattotal = [0 for i in range(n_gt)]
    aa = [0 for i in range(n_gt)]
    for tco in tepixco:
        x = tco[0]
        y = tco[1]
        cattotal[gtmap[x, y] - 1] += 1
        if gtmap[x, y] == 0:
            raise Exception("test pixco Error!")
        if segmap[x, y] == gtmap[x, y]:
            corr += 1
            catcorr[gtmap[x, y] - 1] += 1
        resmat[segmap[x, y] - 1, gtmap[x, y] - 1] += 1
    print("Res AA: ")
    for i in range(n_gt):
        aa[i] = catcorr[i] / cattotal[i]
        print(aa[i])

    po = np.sum(np.diag(resmat)) / n_t
    pe1 = np.sum(resmat, axis=1)
    pe2 = np.sum(resmat, axis=0)
    pe = np.dot(pe1, pe2) / (n_t * n_t)
    ka = (po - pe) / (1 - pe)
    return corr / n_t, np.mean(aa), ka, aa


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
    trESrep = ESrep[trsupidx]
    trxys = []
    for tridx in trsupidx:
        trxys.append(supxys[tridx])

    teESrep = ESrep[tesupidx]
    texys = []
    for teidx in tesupidx:
        texys.append(supxys[teidx])

    if method == 'svm':
        svm = SVC(C=C, gamma=gamma, decision_function_shape='ovo', probability=True)
        svm.fit(trESrep, trsuplabel)
        t1 = time.time()
        predSuplabel = svm.predict(teESrep)
        supproba = svm.predict_proba(teESrep)
        t2 = time.time()
        inference_t = t2 - t1
        print("inference time", inference_t)

    supprobaidx = np.argsort(-supproba, axis=1)
    shighest = supprobaidx[:, 0]
    ssecond = supprobaidx[:, 1]
    predsupcer = np.zeros(shape=np.shape(supproba)[0])
    print(np.max(supproba, axis=1))
    for i, h in enumerate(shighest):
        predsupcer[i] = (supproba[i, h] - supproba[i, ssecond[i]])

    res_map, supcermap = milab2segmap_sup(trsuplabel, trxys, predSuplabel, predsupcer, texys, length, width)
    return res_map, supcermap, tr_pixcos, te_pixcos, rest_pixcos, inference_t


def supResFusion(resmap, cermap, resmap1, cermap1, tepixcos):
    rows, cols = np.shape(resmap)
    newresmap = np.zeros(shape=(rows, cols))
    newcermap = np.zeros(shape=(rows, cols))
    for co in tepixcos:
        x = co[0]
        y = co[1]
        if cermap[x, y] >= cermap1[x, y]:
            newresmap[x, y] = resmap[x, y]
            newcermap[x, y] = cermap[x, y]
        else:
            newresmap[x, y] = resmap1[x, y]
            newcermap[x, y] = cermap1[x, y]
    return newresmap, newcermap
