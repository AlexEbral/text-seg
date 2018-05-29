from nltk.corpus import words
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import os
from multiprocessing.dummy import Pool as ThreadPool
import PIL.ImageOps
import cv2
import string
import time
from alphabet_ru_short import ALPHABET_DICT_COLORED, alphabet
from convert_to_tf_labels import convert_to_tf_mask, convert_to_tf_mask_bin
from get_fonts_list import fonts_list
from utils import get_russian_words
from preprocess import canny_preprocess

FONTS_PATH, bad_fonts_list = fonts_list(russian=True)
FONTS_PATH_LEN = len(FONTS_PATH)
WORDS = get_russian_words()
basewidth = 800

root = ''


from rotate_3d.image_transformer import ImageTransformer


def gen_random_phrase(phrase_len=10, split_cnt=3):
    punctuation = [',', ';', '!', ':', '.']
    words_list = random.sample(WORDS, phrase_len)
    capitalization = random.randint(1, 3)
    if capitalization == 2:
        words_list[0] = words_list[0].capitalize()

    words_count = len(words_list)
    punc_len = len(punctuation)
    # print(words_list)
    # print(words_count)
    phrase = ''
    if words_count > split_cnt:
        for ind, word in enumerate(words_list):
            # is_punc = random.randint(1, 2) == 2
            is_punc = True
            splitter = ''
            if is_punc:
                punc_ind = random.randint(0, punc_len - 1)
                splitter = punctuation[punc_ind]

            # is_num = random.randint(1, 2) == 2
            is_num = True
            num = ''
            if is_num:
                num = str(random.randint(111111, 999999))
            phrase += words_list[ind] + splitter + ' ' + num + ' '
            if (ind + 1) % split_cnt == 0:
                phrase += '\n'
        # print(phrase)
    else:
        phrase = ' '.join(words_list)
    if capitalization == 3:
        return phrase.upper()
    return phrase


def gen_brightness(mean_value):
    return random.randint(0, 30) if mean_value > 128 else random.randint(225, 255)
    # return int(mean_value * 1.5) % 255 if mean_value > 128 else int(mean_value * 0.5)
    # return int(mean_value * 1.2) % 255


def gen_phrase_mask(im, phrase_list):
    len_phrase_list = len(phrase_list)
    height_chunks_num = 2
    width_chunks_num = int(len_phrase_list/height_chunks_num)
    mask_width, mask_height = im.size
    out = im.convert('RGBA')

    height_chunk_size = int(mask_height/height_chunks_num)
    width_chunk_size = int(mask_width/width_chunks_num)
    # print(mask_width, mask_height)
    points_list = []
    for i in range(1, height_chunks_num+1):
        for j in range(1, width_chunks_num+1):
            points_list.append(( (j-1)*width_chunk_size, (i-1)*height_chunk_size))
            # print((i-1)*height_chunk_size, (j-1)*width_chunk_size)

    txt_for_image = Image.new('RGBA', im.size, (255, 255, 255, 255))
    txt_for_tf = Image.new('RGBA', im.size, (255, 255, 255, 255))

    brightness = []
    for ind in range(4):
        upper_left_x, upper_left_y = points_list[ind]
        bottom_right_x = upper_left_x + width_chunk_size
        bottom_right_y = upper_left_y + height_chunk_size

        # print(upper_left_x, upper_left_y, bottom_right_x, bottom_right_y)
        cropped = out.crop((upper_left_x, upper_left_y, bottom_right_x, bottom_right_y))
        arr = np.array(cropped)
        # cropped.show()
        brightness.append((int(np.mean(arr[:, :, 0])),
                           int(np.mean(arr[:, :, 1])),
                           int(np.mean(arr[:, :, 2])))
                        )
    # print(brightness)

    for ind, phrase in enumerate(phrase_list):
        text_size = random.randint(17, 30)

        # text_fnt = 'Arial.ttf'
        text_fnt = FONTS_PATH[random.randint(0, FONTS_PATH_LEN-1)]
        # print(text_fnt, phrase)
        fnt = ImageFont.truetype(text_fnt, text_size)

        txt = Image.new('RGBA', (width_chunk_size, height_chunk_size),
                        (0, 0, 0, 0))

        d = ImageDraw.Draw(txt)
        d.fontmode = "1"

        # text_position = (0, 0)
        text_position = (random.randint(int(width_chunk_size/4), int(width_chunk_size/2)),
                         random.randint(int(height_chunk_size/4), int(height_chunk_size/2)))

        w, h = text_position
        curr_width = w
        curr_height = h
        for i, char in enumerate(phrase):
            if char == "\n":
                # break
                curr_width = w
                curr_height += text_size
            # print(char.rjust(i + 1))
            # print(text_position)
            # d.text((w + i*text_size, h), char.rjust(i + 1), font=fnt, fill=tuple(text_color))
            diff_w = 0
            if char == ' ':
                diff_w = 10
            else:
                diff_w, diff_h = fnt.getsize(char)

            # char_num = ord(char)
            # CHARS.append(char_num)
            # print(char, char_num)
            if char != ' ' and char != '\n':
                text_color = ALPHABET_DICT_COLORED[char.lower()] + (255,)
                d.text((curr_width, curr_height), char, font=fnt, fill=text_color)
            if char != '\n':
                curr_width += diff_w
            # fnt.getmask(char).show()


        mean_r, mean_g, mean_b = brightness[ind]

        text_color = (gen_brightness(mean_r), # R
                      gen_brightness(mean_g), # G
                      gen_brightness(mean_b), # B
                      255) # A

        txt_copy = txt.copy()
        pixel_data = txt_copy.load()

        for y in range(txt_copy.size[1]):
            for x in range(txt_copy.size[0]):
                if pixel_data[x, y][3] != 0:
                    pixel_data[x, y] = text_color

        angle_range = (-20, 20)
        rand_theta = random.randint(*angle_range)
        rand_phi = random.randint(*angle_range)
        rand_gamma = random.randint(*angle_range)

        img_transformer = ImageTransformer(txt)
        txt = img_transformer.rotate_along_axis(theta=rand_theta,
                                                phi=rand_phi,
                                                gamma=rand_gamma)
        txt = Image.fromarray(txt)

        img_transformer = ImageTransformer(txt_copy)
        txt_copy = img_transformer.rotate_along_axis(theta=rand_theta,
                                                phi=rand_phi,
                                                gamma=rand_gamma)
        txt_copy = Image.fromarray(txt_copy)
        upper_left_pnt = points_list[ind]
        txt_for_image.paste(txt_copy, upper_left_pnt)
        txt_for_tf.paste(txt, upper_left_pnt)

    out = Image.alpha_composite(out, txt_for_image)

    return out.convert('L'), txt_for_image.convert('RGB'), txt_for_tf.convert('RGB')


def gen_alphabet_sample():
    alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789,;!:.-'
    alphabet_list = []
    for i in range(4):
        alphabet_list.extend(random.sample(alphabet, 25))
        alphabet_list.append('\n')
        alphabet_list.extend(random.sample(alphabet, 25))
        alphabet_list.append('\n')

    # print(alphabet_list)
    return ''.join(alphabet_list)


def gen_pics_and_masks(file_path, ind, binary=True, random_symbols=False, test=True):
    try:
        if not random_symbols:
            phrase_list = [gen_random_phrase(10, 2) for x in np.arange(4)]
        else:
            phrase_list = [gen_alphabet_sample(), gen_alphabet_sample(), gen_alphabet_sample(), gen_alphabet_sample()]
        im = Image.open(file_path)
        wpercent = (basewidth/float(im.size[0]))
        hsize = int((float(im.size[1])*float(wpercent)))
        im = im.resize((basewidth,hsize), Image.ANTIALIAS)

        img, mask, tf_mask = gen_phrase_mask(im, phrase_list)

        filename = os.path.splitext(os.path.basename(file_path))[0].replace(' ', '_')
        img_path = os.path.join('pics', filename + '_' + str(ind) + '.png')
        # mask_path = os.path.join(root + 'mask', filename + '_' + str(ind) + '.png')
        tf_mask_path = os.path.join('mask', filename + '_' + str(ind) + '.png')
        # img.resize((640, 360)).save(img_path)

        # tf_mask.show()
        arr = np.array(tf_mask)
        if not binary:
            tf_mask_arr_2d = convert_to_tf_mask(arr)
        else:
            tf_mask_arr_2d = convert_to_tf_mask_bin(arr)


        if test:
            img.show()
            # mask.show()
            # tf_mask = Image.fromarray(tf_mask_arr_2d)
            # tf_mask.show()

        # img.save(img_path, quality=60, optimize=True)
        img.save(root + img_path)
        img = np.array(img)
        selection_mask, img = canny_preprocess(img)
        cv2.imwrite(root + os.path.join('pics', filename + '_' + str(ind) + '_prep.png'), img)

        # mask.save(mask_path)
        tf_mask.save(root + tf_mask_path)
        # tf_mask = cv2.bitwise_and(tf_mask_arr_2d, tf_mask_arr_2d, mask=selection_mask)
        # cv2.imwrite(root + tf_mask_path, tf_mask)

        print(img_path, tf_mask_path)
        # return img_path, tf_mask_path
        return os.path.splitext(os.path.basename(img_path))[0]
    except Exception as e:
        print('exception!!!!!', repr(e))
        return ''


def save_to_file_old(file_path, paths):
    with open(file_path, 'w') as f:
        for img_path, tf_mask_path in paths:
            if img_path == '':
                continue
            new_str = img_path + '\t' + tf_mask_path + '\n'
            f.write(new_str)


def save_to_file(file_path, paths):
    with open(file_path, 'w') as f:
        for img_path in paths:
            if img_path == '':
                continue
            new_str = img_path + '\n'
            f.write(new_str)

def save_splits(result_paths, percent=0.8):
    res_len = len(result_paths)
    train_path = root + 'splits/train.txt'
    train_paths = result_paths[:int(res_len*percent)]
    save_to_file(train_path, train_paths)

    test_path = root + 'splits/test.txt'
    test_paths = result_paths[-int(res_len*(1 - percent)):]
    save_to_file(test_path, test_paths)


def generate(test):
    if not test:
        path_to_backgrounds = '/mnt/hgfs/xubuntu64/wallpapers/'

        pool_func_args = []
        for dir, _, pics in os.walk(path_to_backgrounds):
            for pic in pics:
                file_path = os.path.join(dir, pic)
                for ind in range(5):
                    pool_func_args.append((file_path, ind))

        pool = ThreadPool(4)
        start_time = time.time()
        results = pool.starmap(gen_pics_and_masks, pool_func_args)
        runtime = time.time() - start_time
        print(runtime)
        save_splits(results)

        pool.close()
        pool.join()
    else:
        img = gen_pics_and_masks('test/for_dipl/pic (390).jpg', 0)
    # print(np.random.randn(1, 2))
    # print(len(words.words()))


if __name__ == "__main__":
    ts = time.time()
    generate(test=True)
    # phrase_list = [gen_random_phrase(random.randint(10, 50)) for x in np.arange(4)]
    # print(phrase_list)
    print(time.time() - ts)




