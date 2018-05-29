import cv2
import numpy as np


def convert_to_mask_bin(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    m = np.all(arr_3d == np.array((0, 0, 0)).reshape(1, 1, 3), axis=2)
    arr_2d[m] = 0
    m = np.all(arr_3d != np.array((0, 0, 0)).reshape(1, 1, 3), axis=2)
    arr_2d[m] = 255
    return arr_2d

# img = cv2.imread('./10.png')
# mask = convert_to_mask_bin(img)
# cv2.imwrite('./10_mask.png', mask)


def get_russian_words():
    with open('singular.txt', 'r') as file:
        return list(map(lambda word: word[:-1], file.readlines()))