from preprocess import canny_preprocess
import cv2
import os


dir = ''
for dir, _, pics in os.walk(dir):
    for pic in pics:
        file_path = os.path.join(dir, pic)
        prep_file_path = file_path.split('.')[0] + '_prep.jpg'
        print(file_path)
        img = cv2.imread(file_path, 0)
        mask, img = canny_preprocess(img)
        cv2.imwrite(prep_file_path, img)
