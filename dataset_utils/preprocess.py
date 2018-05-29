import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

kernel = np.ones((5, 5), np.uint8)


def canny_preprocess(img):
    edges = cv2.Canny(img, 0, 500)
    mask = cv2.dilate(edges, kernel, iterations=1)
    # plt.imshow(mask, cmap='gray')
    return mask, cv2.bitwise_and(img, img, mask=mask)
    # edges = cv2.bitwise_not(edges)
    # return edges






