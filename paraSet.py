import numpy as np


def HMSparaSet(dataset):
    maindir = ""  ####  the path of datasets  ####
    if dataset == 'IP':
        filedir = "indian_pines_corrected.mat"
        gtdir = "indian_pines_gt.mat"
        K = [256, 384, 768]  # [256, 384, 768]
        matname = 'indian_pines_corrected'
        gtname = 'indian_pines_gt'
        gt_n = 16
        tr_num = 1024
        tr_ratio = 0.01
        C = 10  # 10
        gamma = 1e-3
        n1 = 3  # 3
        n2 = 1
        fratio = 4  # 1
        ssr = 0.125
        seed = [0, 7, 2, 8, 12]
        return maindir, filedir, gtdir, K, matname, gtname, gt_n, tr_num, tr_ratio, C, gamma, n1, n2, fratio, ssr, seed
    if dataset == 'SA':
        filedir = "salinas_corrected.mat"
        gtdir = "salinas_gt.mat"
        K = [384, 512, 768]
        matname = 'salinas_corrected'
        gtname = 'salinas_gt'
        gt_n = 16
        tr_num = 1024
        tr_ratio = 0.003
        C = 10  # 100
        gamma = 1e-3
        n1 = 3  # 3
        n2 = 1
        fratio = 4  # 4
        ssr = 0.125
        seed = [0, 7, 2, 8, 12]
        return maindir, filedir, gtdir, K, matname, gtname, gt_n, tr_num, tr_ratio, C, gamma, n1, n2, fratio, ssr, seed
    if dataset == 'PU':
        maindir = "D:\\资料\\Datas\\Hyperspectral_Data_Opensource\\PaviaU\\"
        filedir = "PaviaU.mat"
        gtdir = "PaviaU_gt.mat"
        K = [512, 768, 1024]
        matname = 'paviaU'
        gtname = 'paviaU_gt'
        gt_n = 9
        tr_num = 1024
        tr_ratio = 0.001
        C = 10  # 100
        gamma = 1e-3
        n1 = 3  # 3
        n2 = 1
        fratio = 4  # 4
        ssr = 0.125
        seed = [0, 7, 2, 8, 12]  # [0, 7, 2, 8, 12]
        return maindir, filedir, gtdir, K, matname, gtname, gt_n, tr_num, tr_ratio, C, gamma, n1, n2, fratio, ssr, seed
    if dataset == 'UH':
        maindir = ".\\datasets\\UH\\"
        filedir = "houston.mat"
        gtdir = "houston_gt.mat"
        matname = 'houston'
        gtname = ['houston_gt_tr', 'houston_gt_te']
        gt_n = 15
        tr_num = 1024
        tr_ratio = 0.001
        K = [2048, 4096, 8192]
        C = 10
        gamma = 1e-3
        n1 = 3
        n2 = 1
        fratio = 4  # 4
        ssr = 0.125
        seed = [0, 7, 2, 8, 12]
        return maindir, filedir, gtdir, K, matname, gtname, gt_n, tr_num, tr_ratio, C, gamma, n1, n2, fratio, ssr, seed
