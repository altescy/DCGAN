#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 22:11:56 2017

@author: altescy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from chainer import Variable, serializers

from DCGAN import Generator
from utility import clip




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-n', '--num', type=int, default=25)
    parser.add_argument('-o', '--out', type=str, default='tmp.png')
    args = parser.parse_args()
    
    genfile = args.model
    n_img = args.num
    
    nz = 100
    gen = Generator(nz)
    serializers.load_npz(genfile, gen)
    
    x = gen(Variable(np.random.uniform(-1, 1, (n_img, nz)).astype(np.float32)))
    x = ((clip(x.data) + 1) / 2).transpose(0, 2, 3, 1)
    
    tmp = np.floor(np.sqrt(n_img))
    nh, nw = (1, n_img) if n_img <= 5 else (tmp, np.ceil(n_img / tmp))
    plt.figure(figsize=(nw, nh))
    for i in range(n_img):
        plt.subplot(nh, nw, i + 1)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.imshow(x[i])
    plt.savefig(args.out)
    