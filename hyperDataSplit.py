import numpy as np


def hyperdataSplit(gt_img, gt_n, ratio, seed, dataset):
    gt_pixcos = [[] for _ in range(gt_n)]
    if dataset == 'UH':
        tr_pixcos = []
        te_pixcos = []
        rest_pixcos = []
        gt_img_tr = gt_img[0]
        gt_img_te = gt_img[1]
        rows, cols = np.shape(gt_img_tr)
        for i in range(rows):
            for j in range(cols):
                if gt_img_tr[i, j] > 0:
                    tr_pixcos.append([i, j])
                elif gt_img_te[i, j] > 0:
                    te_pixcos.append([i, j])
                else:
                    rest_pixcos.append([i, j])
        return tr_pixcos, te_pixcos, rest_pixcos

    rows, cols = np.shape(gt_img)

    for i in range(rows):
        for j in range(cols):
            if gt_img[i, j] > 0:
                gt_pixcos[gt_img[i, j] - 1].append([i, j])

    tr_pixcos = []
    te_pixcos = []
    rest_pixcos = []
    print("tr num: ***********")
    for i in range(gt_n):
        thiscat_n = len(gt_pixcos[i])
        if 0 < ratio < 1:
            tr_num = np.round(ratio * thiscat_n).astype('uint32')
            tr_num = max(2, tr_num)
        else:
            tr_num = ratio
        print(thiscat_n, "---", tr_num)
        np.random.seed(seed)
        perm = np.random.permutation(thiscat_n)
        tr_idxes = perm[0:tr_num]
        te_idxes = perm[tr_num:thiscat_n]
        for tridx in tr_idxes:
            tr_pixcos.append(gt_pixcos[i][tridx])
        for teidx in te_idxes:
            te_pixcos.append(gt_pixcos[i][teidx])
    for i in range(rows):
        for j in range(cols):
            if gt_img[i, j] == 0:
                rest_pixcos.append([i, j])
    return tr_pixcos, te_pixcos, rest_pixcos


class hyper_Data_Seg(object):
    def __init__(self, train_num, train_ratio, model='fixed', tr_array=[]):
        self.train_num = train_num
        self.train_ratio = train_ratio
        self.model = model
        self.tr_array = tr_array

    def arg_apply(self, hyper_img, gt):
        length, width = np.shape(gt)
        band = np.shape(hyper_img)[2]
        total = length * width
        reshaped_gt = gt.reshape(total)
        pixels = np.mgrid[0:length, 0:width].reshape(2, -1).T

        index_gt = np.argsort(reshaped_gt)
        sorted_gt = reshaped_gt[index_gt]
        sorted_pix = pixels[index_gt]
        labeled_index = 0
        while (sorted_gt[labeled_index] == 0):
            labeled_index += 1
        sorted_gt = sorted_gt[labeled_index:]
        sorted_pix = sorted_pix[labeled_index:]
        hyper_index = []  # 标记各类光谱的起止区域
        start_index = 0
        end_index = 0
        label_now = sorted_gt[0]
        for i in range(sorted_gt.shape[0]):
            if (label_now != sorted_gt[i]):
                end_index = i
                hyper_index.append([start_index, end_index])
                start_index = end_index
            label_now = sorted_gt[i]
        hyper_index.append([start_index, sorted_gt.shape[0]])
        count = 0
        _init_pixs = []
        _pool_pixs = []
        for index in hyper_index:
            if (self.model == 'fixed'):
                ratio_num = np.round(self.train_ratio * (index[1] - index[0])).astype('uint32')
                size_num = min(self.train_num, ratio_num)
                size_num = max(1, size_num)
            elif (self.model == 'array'):
                size_num = self.tr_array[count]
                count += 1
            _indices = np.random.randint(index[0], index[1], size=(size_num),
                                         dtype='int32').astype('uint16')
            ind_set = set(_indices)
            ind_set_ = set(i for i in range(index[0], index[1])) - ind_set
            _indices_ = np.array([i for i in ind_set_])
            if (len(_init_pixs) == 0):
                _init_pixs = sorted_pix[_indices]
                _pool_pixs = sorted_pix[_indices_]
            else:
                _init_pixs = np.concatenate((_init_pixs, sorted_pix[_indices]))
                _pool_pixs = np.concatenate((_pool_pixs, sorted_pix[_indices_]))
        return _init_pixs, _pool_pixs

    def apply(self, hyper_img, gt):
        length, width = np.shape(gt)
        band = np.shape(hyper_img)[2]
        total = length * width
        reshaped_gt = gt.reshape(total)
        pixels = np.mgrid[0:length, 0:width].reshape(2, -1).T

        index_gt = np.argsort(reshaped_gt)
        sorted_gt = reshaped_gt[index_gt]
        sorted_pix = pixels[index_gt]
        labeled_index = 0
        while (sorted_gt[labeled_index] == 0):
            labeled_index += 1
        sorted_gt = sorted_gt[labeled_index:]
        sorted_pix = sorted_pix[labeled_index:]
        hyper_index = []  # 标记各类光谱的起止区域
        hyper_data = []  # 存储带标签的光谱数据
        start_index = 0
        end_index = 0
        label_now = sorted_gt[0]
        for i in range(sorted_gt.shape[0]):
            hyper_data.append(hyper_img[sorted_pix[i][0], sorted_pix[i][1]])
            if (label_now != sorted_gt[i]):
                end_index = i
                hyper_index.append([start_index, end_index])
                start_index = end_index
            label_now = sorted_gt[i]
        hyper_index.append([start_index, sorted_gt.shape[0]])
        hyper_data = np.array(hyper_data)
        train_data = []
        test_data = []
        train_label = []
        test_label = []
        test_label_ind = []
        test_pixels = []
        start = 0
        end = 0
        count = 0
        for index in hyper_index:
            if (self.model == 'fixed'):
                ratio_num = np.round(self.train_ratio * (index[1] - index[0])).astype('uint32')
                size_num = min(self.train_num, ratio_num)
                size_num = max(1, size_num)
            elif (self.model == 'array'):
                size_num = self.tr_array[count]
                count += 1
            indices = np.random.randint(index[0], index[1], size=(size_num),
                                        dtype='int32').astype('uint16')
            # indices = [i for i in range(index[0], index[1], np.round((index[1] - index[0])/size_num).astype('int32'))]
            ind_set = set(indices)
            ind_set_ = set(i for i in range(index[0], index[1])) - ind_set
            indices_ = np.array([i for i in ind_set_])
            if (len(train_data) == 0):
                train_data = hyper_data[np.array(indices)]
                train_label = sorted_gt[indices]
                test_data = hyper_data[indices_]
                test_label = sorted_gt[indices_]
                test_pixels = sorted_pix[indices_]
            else:
                train_data = np.concatenate((train_data, hyper_data[indices]), axis=0)
                train_label = np.concatenate((train_label, sorted_gt[indices]), axis=0)
                test_data = np.concatenate((test_data, hyper_data[indices_]), axis=0)
                test_label = np.concatenate((test_label, sorted_gt[indices_]), axis=0)
                test_pixels = np.concatenate((test_pixels, sorted_pix[indices_]), axis=0)
            end = len(test_label)
            test_label_ind.append([start, end])
            start = end
        return train_data, train_label, test_data, test_label, test_label_ind, test_pixels
