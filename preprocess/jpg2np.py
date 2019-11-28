import PIL.Image as pilimg
import numpy as np
import torch
import os
import glob
import csv
import torchvision
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
 

def main(img_dir, npy_out_dir):
    if not os.path.exists(npy_out_dir):
        os.makedirs(npy_out_dir)

    font_lists = next(os.walk(img_dir))[1]
    
    for i in range(len(font_lists)):
        font_dir = os.path.join(img_dir, font_lists[i])
        images = glob.glob1(font_dir, '*.jpg')

        for img in images:
            im = pilimg.open(os.path.join(font_dir, img))
            im = np.array(im)

            combined = (im, i)
            
            f_out = os.path.join(npy_out_dir, '{}_{}'.format(i, img.split(".")[0]))
            np.save(f_out, combined, allow_pickle=True)
            print("saved: {}".format(img))

        print("Done for font: {}".format(font_lists[i]))



if __name__ == '__main__':
    img_dir = '../datasets/img_dir'
    npy_out_dir = '../datasets/npy_dir'
    main(img_dir, npy_out_dir)
