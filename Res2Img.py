import numpy as np
import cv2


def color_convert_bgr_PaviaU(input, len, wid):
    c_gt = np.zeros(shape=(len, wid, 3), dtype='uint8')
    for i in range(len):
        for j in range(wid):
            if input[i, j] == 1:
                c_gt[i, j] = [192, 192, 192]
            elif input[i, j] == 2:
                c_gt[i, j] = [0, 255, 0]
            elif input[i, j] == 3:
                c_gt[i, j] = [255, 255, 0]
            elif input[i, j] == 4:
                c_gt[i, j] = [34, 139, 34]
            elif input[i, j] == 5:
                c_gt[i, j] = [214, 112, 218]
            elif input[i, j] == 6:
                c_gt[i, j] = [20, 97, 199]
            elif input[i, j] == 7:
                c_gt[i, j] = [240, 32, 160]
            elif input[i, j] == 8:
                c_gt[i, j] = [0, 0, 255]
            elif input[i, j] == 9:
                c_gt[i, j] = [0, 255, 255]
    return c_gt


def color_convert_bgr_KSC(input, len, wid):
    c_gt = np.zeros(shape=(len, wid, 3), dtype='uint8')
    for i in range(len):
        for j in range(wid):
            if input[i, j] == 1:
                c_gt[i, j] = [0, 255, 0]
            elif input[i, j] == 2:
                c_gt[i, j] = [214, 112, 218]
            elif input[i, j] == 3:
                c_gt[i, j] = [30, 105, 210]
            elif input[i, j] == 4:
                c_gt[i, j] = [34, 34, 178]
            elif input[i, j] == 5:
                c_gt[i, j] = [15, 94, 56]
            elif input[i, j] == 6:
                c_gt[i, j] = [18, 38, 94]
            elif input[i, j] == 7:
                c_gt[i, j] = [135, 138, 128]
            elif input[i, j] == 8:
                c_gt[i, j] = [105, 128, 128]
            elif input[i, j] == 9:
                c_gt[i, j] = [205, 235, 255]
            elif input[i, j] == 10:
                c_gt[i, j] = [85, 142, 235]
            elif input[i, j] == 11:
                c_gt[i, j] = [255, 255, 0]
            elif input[i, j] == 12:
                c_gt[i, j] = [84, 46, 8]
            elif input[i, j] == 13:
                c_gt[i, j] = [255, 105, 65]
    return c_gt


def color_convert_bgr_Indian(input, len, wid):
    c_gt = np.zeros(shape=(len, wid, 3), dtype='uint8')
    for i in range(len):
        for j in range(wid):
            if input[i, j] == 1:
                c_gt[i, j] = [192, 192, 192]
            elif input[i, j] == 2:
                c_gt[i, j] = [105, 128, 112]
            elif input[i, j] == 3:
                c_gt[i, j] = [255, 255, 255]
            elif input[i, j] == 4:
                c_gt[i, j] = [84, 46, 8]
            elif input[i, j] == 5:
                c_gt[i, j] = [201, 230, 252]
            elif input[i, j] == 6:
                c_gt[i, j] = [0, 0, 255]
            elif input[i, j] == 7:
                c_gt[i, j] = [135, 138, 128]
            elif input[i, j] == 8:
                c_gt[i, j] = [31, 102, 156]
            elif input[i, j] == 9:
                c_gt[i, j] = [96, 48, 176]
            elif input[i, j] == 10:
                c_gt[i, j] = [255, 0, 255]
            elif input[i, j] == 11:
                c_gt[i, j] = [0, 255, 0]
            elif input[i, j] == 12:
                c_gt[i, j] = [0, 255, 255]
            elif input[i, j] == 13:
                c_gt[i, j] = [20, 48, 128]
            elif input[i, j] == 14:
                c_gt[i, j] = [42, 42, 128]
            elif input[i, j] == 15:
                c_gt[i, j] = [64, 145, 61]
            elif input[i, j] == 16:
                c_gt[i, j] = [255, 255, 0]
    return c_gt


def addTrPix(tr_pixcos, gt_img, res):
    for pix in tr_pixcos:
        x = pix[0]
        y = pix[1]
        res[x, y] = gt_img[x, y]
    return res


def res2img(dataset, tr_ratio, res):
    maindir = '' + str(dataset) + '-' + str(tr_ratio) + '-'
    length, width = np.shape(res)
    if dataset == 'PU':
        img = color_convert_bgr_PaviaU(res, length, width)
    else:
        img = color_convert_bgr_Indian(res, length, width)
    cv2.imencode('.jpg', img)[1].tofile(maindir + 'res_img' + '.jpg')


def gt2img(dataset, res):
    maindir = '' + str(dataset) + '-'
    length, width = np.shape(res)
    if dataset == 'PU':
        img = color_convert_bgr_PaviaU(res, length, width)
    else:
        img = color_convert_bgr_Indian(res, length, width)
    cv2.imencode('.jpg', img)[1].tofile(maindir + 'gt_img' + '.jpg')
