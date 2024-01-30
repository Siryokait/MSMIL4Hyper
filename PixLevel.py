import numpy as np


def pixco2pixbag(tr_pixco, te_pixco, rest_pixco, Hsi, gt_map, rows, cols):
    instance = []
    bag_ids = []
    tr_n = np.shape(tr_pixco)[0]
    te_n = np.shape(te_pixco)[0]
    tr_idx = []
    te_idx = []
    rest_idx = []
    labels = []
    for i, co in enumerate(tr_pixco):
        x = co[0]
        y = co[1]
        tr_idx.append(i)
        labels.append(gt_map[x, y])
        if x > 0:
            instance.append(Hsi[x - 1, y])
            bag_ids.append(i)
            if y > 0:
                instance.append(Hsi[x - 1, y - 1])
                bag_ids.append(i)
            if y < cols - 1:
                instance.append(Hsi[x - 1, y + 1])
                bag_ids.append(i)
        if x < rows - 1:
            instance.append(Hsi[x + 1, y])
            bag_ids.append(i)
            if y > 0:
                instance.append(Hsi[x + 1, y - 1])
                bag_ids.append(i)
            if y < cols - 1:
                instance.append(Hsi[x + 1, y + 1])
                bag_ids.append(i)
        if y > 0:
            instance.append(Hsi[x, y - 1])
            bag_ids.append(i)
        if y < cols - 1:
            instance.append(Hsi[x, y + 1])
            bag_ids.append(i)

    for i, co in enumerate(te_pixco):
        x = co[0]
        y = co[1]
        te_idx.append(i + tr_n)
        labels.append(gt_map[x, y])
        if x > 0:
            instance.append(Hsi[x - 1, y])
            bag_ids.append(i + tr_n)
            if y > 0:
                instance.append(Hsi[x - 1, y - 1])
                bag_ids.append(i + tr_n)
            if y < cols - 1:
                instance.append(Hsi[x - 1, y + 1])
                bag_ids.append(i + tr_n)
        if x < rows - 1:
            instance.append(Hsi[x + 1, y])
            bag_ids.append(i + tr_n)
            if y > 0:
                instance.append(Hsi[x + 1, y - 1])
                bag_ids.append(i + tr_n)
            if y < cols - 1:
                instance.append(Hsi[x + 1, y + 1])
                bag_ids.append(i + tr_n)
        if y > 0:
            instance.append(Hsi[x, y - 1])
            bag_ids.append(i + tr_n)
        if y < cols - 1:
            instance.append(Hsi[x, y + 1])
            bag_ids.append(i + tr_n)

    for i, co in enumerate(rest_pixco):
        x = co[0]
        y = co[1]
        rest_idx.append(i + tr_n + te_n)
        labels.append(gt_map[x, y])
        if x > 0:
            instance.append(Hsi[x - 1, y])
            bag_ids.append(i + tr_n + te_n)
            if y > 0:
                instance.append(Hsi[x - 1, y - 1])
                bag_ids.append(i + tr_n + te_n)
            if y < cols - 1:
                instance.append(Hsi[x - 1, y + 1])
                bag_ids.append(i + tr_n + te_n)
        if x < rows - 1:
            instance.append(Hsi[x + 1, y])
            bag_ids.append(i + tr_n + te_n)
            if y > 0:
                instance.append(Hsi[x + 1, y - 1])
                bag_ids.append(i + tr_n + te_n)
            if y < cols - 1:
                instance.append(Hsi[x + 1, y + 1])
                bag_ids.append(i + tr_n + te_n)
        if y > 0:
            instance.append(Hsi[x, y - 1])
            bag_ids.append(i + tr_n + te_n)
        if y < cols - 1:
            instance.append(Hsi[x, y + 1])
            bag_ids.append(i + tr_n + te_n)
    return np.array(instance), np.array(bag_ids), np.array(tr_idx), np.array(te_idx), np.array(rest_idx), np.array(
        labels)


def pixres2resmap(prdres, pixcer, pixcos, rows, cols):
    segmap = np.zeros(shape=(rows, cols), dtype=int)
    cermap = np.zeros(shape=(rows, cols))
    for i, co in enumerate(pixcos):
        t_res = prdres[i]
        x = co[0]
        y = co[1]
        segmap[x, y] = t_res
        cermap[x, y] = pixcer[i]
    return segmap, cermap


def spatCer(cermap, resmap, tepixcos, ssr):
    rows, cols = np.shape(cermap)
    ss_ratio = ssr
    sscermap = np.zeros(shape=(rows, cols))
    for i, co in enumerate(tepixcos):
        x = co[0]
        y = co[1]
        t_res = resmap[x, y]
        sscermap[x, y] = cermap[x, y]
        if x > 0:
            if resmap[x - 1, y] == t_res:
                sscermap[x, y] += ss_ratio * cermap[x - 1, y]
            else:
                sscermap[x, y] -= ss_ratio * cermap[x - 1, y]
            if y > 0:
                if resmap[x - 1, y - 1] == t_res:
                    sscermap[x, y] += ss_ratio * cermap[x - 1, y - 1]
                else:
                    sscermap[x, y] -= ss_ratio * cermap[x - 1, y - 1]
            if y < cols - 1:
                if resmap[x - 1, y + 1] == t_res:
                    sscermap[x, y] += ss_ratio * cermap[x - 1, y + 1]
                else:
                    sscermap[x, y] -= ss_ratio * cermap[x - 1, y + 1]
        if x < rows - 1:
            if resmap[x + 1, y] == t_res:
                sscermap[x, y] += ss_ratio * cermap[x + 1, y]
            else:
                sscermap[x, y] -= ss_ratio * cermap[x + 1, y]
            if y > 0:
                if resmap[x + 1, y - 1] == t_res:
                    sscermap[x, y] += ss_ratio * cermap[x + 1, y - 1]
                else:
                    sscermap[x, y] -= ss_ratio * cermap[x + 1, y - 1]
            if y < cols - 1:
                if resmap[x + 1, y + 1] == t_res:
                    sscermap[x, y] += ss_ratio * cermap[x + 1, y + 1]
                else:
                    sscermap[x, y] -= ss_ratio * cermap[x + 1, y + 1]
        if y > 0:
            if resmap[x, y - 1] == t_res:
                sscermap[x, y] += ss_ratio * cermap[x, y - 1]
            else:
                sscermap[x, y] -= ss_ratio * cermap[x, y - 1]
        if y < cols - 1:
            if resmap[x, y + 1] == t_res:
                sscermap[x, y] += ss_ratio * cermap[x, y + 1]
            else:
                sscermap[x, y] -= ss_ratio * cermap[x, y + 1]
    return sscermap


def resmapFusion(pixresmap, pixcermap, supresmap, supcermap, tepixcos, fr):
    rows, cols = np.shape(pixresmap)
    Fresmap = np.zeros(shape=(rows, cols), dtype=int)
    Fcermap = np.zeros(shape=(rows, cols), dtype=int)

    for co in tepixcos:
        x = co[0]
        y = co[1]
        pixcer = pixcermap[x, y]
        pixres = pixresmap[x, y]
        supcer = supcermap[x, y]
        supres = supresmap[x, y]
        if pixcer > fr * supcer:
            # print(pixcer, supcer)
            Fresmap[x, y] = pixres
            Fcermap[x, y] = pixcer
        else:
            Fresmap[x, y] = supres
            Fcermap[x, y] = supcer
    return Fresmap, Fcermap
