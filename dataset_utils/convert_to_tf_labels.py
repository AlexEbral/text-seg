from PIL import Image
import numpy as np
import cv2
from alphabet_ru_short import DIC_IND_COLOR

def convert_to_tf_mask_bin(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    m = np.all(arr_3d == np.array((0, 0, 0)).reshape(1, 1, 3), axis=2)
    arr_2d[m] = 0
    m = np.all(arr_3d != np.array((0, 0, 0)).reshape(1, 1, 3), axis=2)
    arr_2d[m] = 1
    return arr_2d


def convert_to_tf_mask(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    for ind, color in DIC_IND_COLOR.items():
        m = np.all(arr_3d == np.array(color).reshape(1, 1, 3), axis=2)
        arr_2d[m] = ind
    return arr_2d


if __name__ == "__main__":
    import os, shutil

    new_label_dir = 'mask_tf/'

    label_dir = 'mask/'
    label_files = os.listdir(label_dir)

    for l_f in label_files:
        arr = np.array(Image.open(label_dir + l_f))
        arr_2d = convert_to_tf_mask(arr)
        Image.fromarray(arr_2d).save(new_label_dir + l_f)
