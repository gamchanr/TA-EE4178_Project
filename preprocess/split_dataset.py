import json
import os
import glob
import random
import shutil


def split_dataset(move_numb, A_dir, B_dir):
    entry = glob.glob1(A_dir, '*npy')

    if not os.path.exists(B_dir):
        os.makedirs(B_dir)

    move_target = random.sample(entry, move_numb)

    for target in move_target:
        shutil.move(os.path.join(A_dir, target), os.path.join(B_dir, target))

    A_after = glob.glob1(A_dir, '*npy')
    B_after = glob.glob1(B_dir, '*npy')
    print("Done: Base-data: {}, Moved-data: {}".format(len(A_after), len(B_after)))


if __name__ == '__main__':
    move_numb = 5000
    A_dir = '../datasets/npy_extra'
    B_dir = '../datasets/npy_val'
    split_dataset(move_numb, A_dir, B_dir)
