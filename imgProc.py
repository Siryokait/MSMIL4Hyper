import numpy as np


def difedge(img):
    [rows, cols] = np.shape(img)
    edgeres = np.zeros(shape=(rows, cols))
    for i in range(rows):
        for j in range(cols):
            pixc = img[i, j]
            if i > 0 and img[i - 1, j] != pixc:
                edgeres[i, j] = 1
            elif i < rows - 1 and img[i + 1, j] != pixc:
                edgeres[i, j] = 1
            elif j > 0 and img[i, j - 1] != pixc:
                edgeres[i, j] = 1
            elif j < cols - 1 and img[i, j + 1] != pixc:
                edgeres[i, j] = 1
    return edgeres