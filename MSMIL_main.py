import numpy as np
from scipy.io import loadmat
from paraSet import HMSparaSet
from SupLevel import segmap2acc, wrappedSup
from PixLevel import resmapFusion, spatCer
from Res2Img import res2img, gt2img, addTrPix
import sys
import time
import fire


class expManage():

    def run(self, dataset_name='IP', train_ratio=0.03):
        supload = 0
        drawimg = 0
        dataset = dataset_name
        repeat = 1 if dataset == 'UH' else 5
        method = 'svm'
        maindir, filedir, gtdir, K, matkey, gtkey, gt_n, tr_num, tr_ratio, C, gamma, n1, n2, fratio, ssr, seed = HMSparaSet(
            dataset=dataset)
        tr_ratio = train_ratio
        hyper_data = loadmat(maindir + filedir)[matkey]
        gt_img = loadmat(maindir + gtdir)[gtkey] if dataset != 'UH' else [loadmat(maindir + gtdir)[gtkey[0]],
                                                                          loadmat(maindir + gtdir)[gtkey[1]]]

        length, width, band = np.shape(hyper_data)
        print(str(length) + ", " + str(width) + ", " + str(band))
        hyper_data_d = np.array(hyper_data, dtype='double')

        foa = []
        faa = []
        fka = []

        for i in range(repeat):
            tt1 = time.time()
            res_map, supcermap, tr_pixcos, te_pixcos, rest_pixcos, inf_t = wrappedSup(hyper_data_d, supload, maindir,
                                                                                      K[0],
                                                                                      gt_img, gt_n,
                                                                                      tr_ratio, C, gamma, length, width,
                                                                                      n1, n2,
                                                                                      method, seed[i], dataset)

            res_map1, supcermap1, tr_pixcos1, te_pixcos1, rest_pixcos1, inf_t1 = wrappedSup(hyper_data_d, supload,
                                                                                            maindir,
                                                                                            K[1], gt_img,
                                                                                            gt_n, tr_ratio, C, gamma,
                                                                                            length, width, n1, n2,
                                                                                            method,
                                                                                            seed[i], dataset)

            res_map2, supcermap2, tr_pixcos2, te_pixcos2, rest_pixcos2, inf_t2 = wrappedSup(hyper_data_d, supload,
                                                                                            maindir,
                                                                                            K[2],
                                                                                            gt_img, gt_n, tr_ratio, C,
                                                                                            gamma,
                                                                                            length,
                                                                                            width, n1, n2, method,
                                                                                            seed[i],
                                                                                            dataset)
            tt2 = time.time()
            t1 = tt2 - tt1

            if dataset != 'UH':
                oa, aa, ka2, _ = segmap2acc(res_map, gt_img, te_pixcos)
            else:
                oa, aa, ka2, _ = segmap2acc(res_map, gt_img[1], te_pixcos)
            print("train num: ", np.shape(tr_pixcos)[0])
            print("OA: ", oa)
            print("AA: ", aa)

            # sspixcermap = spatCer(pixcermap, pixresmap, te_pixcos, ssr)
            tt1 = time.time()
            sssupcermap = spatCer(supcermap, res_map, te_pixcos, ssr)
            sssupcermap1 = spatCer(supcermap1, res_map1, te_pixcos, ssr)
            sssupcermap2 = spatCer(supcermap2, res_map2, te_pixcos, ssr)
            # # nSupresmap, nSupcermap = supResFusion(res_map, sssupcermap, res_map1, sssupcermap1, te_pixcos)
            #
            # # Fresmap = resmapFusion(pixresmap, sspixcermap, nSupresmap, nSupcermap, te_pixcos, fratio)
            Fresmap1, Fcermap1 = resmapFusion(res_map1, sssupcermap1, res_map, sssupcermap, te_pixcos, fratio)
            Fresmap, Fcermap = resmapFusion(res_map2, sssupcermap2, Fresmap1, Fcermap1, te_pixcos, fratio)
            tt2 = time.time()
            t2 = tt2 - tt1

            print("Total Time: ", t1 + t2)
            print("Inference Time: ", inf_t + inf_t1 + inf_t2)
            # Fresmap, Fcermap = resmapFusion(res_map1, sssupcermap1, res_map, sssupcermap, te_pixcos, fratio)
            if dataset != 'UH':
                oa, aa, ka, aa1 = segmap2acc(Fresmap, gt_img, te_pixcos)
            else:
                oa, aa, ka, aa1 = segmap2acc(Fresmap, gt_img[1], te_pixcos)
            print("Fusion OA: ", oa)
            print("Fusion AA: ", aa)
            print("Fusion Ka: ", ka)
            foa.append(oa)
            fka.append(ka)
            faa.append(aa1)

            for k, co in enumerate(tr_pixcos):
                co1 = tr_pixcos1[k]
                if co1[0] != co[0] or co1[1] != co[1]:
                    print("tr pix error ----------------")
                    sys.exit(1)

            if drawimg == 1 and i == 0:
                if dataset == 'UH':
                    gt2img(dataset, gt_img[1])
                    res2img(dataset, tr_ratio, Fresmap)
                else:
                    Fresmap = addTrPix(tr_pixcos, gt_img, Fresmap)
                    gt2img(dataset, gt_img)
                    res2img(dataset, tr_ratio, Fresmap)
        print("*****************")
        print("Final OA: ")
        print(np.mean(foa), "-----", np.std(foa))
        print("Final AA: ")
        print(np.mean(faa), "-----", np.std(faa))
        print("Final Ka: ")
        print(np.mean(fka), "-----", np.std(fka))
        print("Final AA detail: ")
        aastd = np.std(faa, axis=0)
        for i, res in enumerate(np.mean(faa, axis=0)):
            print(res, "-----", aastd[i])


if __name__ == '__main__':
    fire.Fire(expManage)
