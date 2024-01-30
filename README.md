# MSMIL for Hyperspectral Classification
The source code of paper "Superpixel-Based Multi-Scale Multi-Instance Learning for Hyperspectral Image Classification".

If you use this code, please cite our work:
```bash
Shiluo Huang, Zheng Liu, Wei Jin, Ying Mu. "Superpixel-based multi-scale multi-instance learning for hyperspectral image classification." Pattern Recognition (2024): 110257.
```
or
```bash
@article{HUANG2024110257,
title = {Superpixel-based multi-scale multi-instance learning for hyperspectral image classification},
journal = {Pattern Recognition},
volume = {149},
pages = {110257},
year = {2024},
issn = {0031-3203},
author = {Shiluo Huang and Zheng Liu and Wei Jin and Ying Mu},
```

We use [HMS](https://github.com/psellcam/Superpixel-Contracted-Graph-Based-Learning-for-Hyperspectral-Image-Classification) for superpixel segmentation.

Then users can try `MSMIL` by running the following command:
```bash
python MSMIL_main.py run --dataset_name IP --train_ratio 0.03
```
