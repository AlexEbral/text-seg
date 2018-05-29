from nltk.corpus import words
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import os
from multiprocessing.dummy import Pool as ThreadPool
import PIL.ImageOps
import cv2
import string
from alphabet import ALPHABET_DICT_COLORED
from convert_to_tf_labels import convert_to_tf_mask
from get_fonts_list import fonts_list

FONTS_PATH, bad_fonts_list = fonts_list()
FONTS_PATH_LEN = len(FONTS_PATH)
WORDS = words.words()
basewidth = 800

from rotate_3d.image_transformer import ImageTransformer


def gen_random_phrase(phrase_len=10, split_cnt=3):
    punctuation = [',', ';', '!', ':', '.']
    words_list = random.sample(WORDS, random.randint(1, phrase_len))
    capitalization = random.randint(1, 3)
    if capitalization == 2:
        words_list[0] = words_list[0].capitalize()

    words_count = len(words_list)
    # print(words_list)
    # print(words_count)
    phrase = ''
    if words_count > split_cnt:
        for ind, word in enumerate(words_list):
            is_punc = random.randint(1, 100) < 20
            splitter = ''
            if is_punc:
                punc_ind = random.randint(0, len(punctuation) - 1)
                splitter = punctuation[punc_ind]
            phrase += words_list[ind] + splitter + ' '
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
        text_fnt = FONTS_PATH[random.randint(0, FONTS_PATH_LEN)]
        fnt = ImageFont.truetype(text_fnt, text_size)

        txt = Image.new('RGBA', (width_chunk_size, height_chunk_size),
                        (255, 255, 255, 0))
        d = ImageDraw.Draw(txt)
        d.fontmode = "1"

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
                text_color = ALPHABET_DICT_COLORED[char] + (255,)
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

        angle_range = (-30, 30)
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

root = '/mnt/hgfs/xubuntu64/text_ds_final/'

def gen_pics_and_masks(file_path, ind):
    try:
        phrase_list = [gen_random_phrase(random.randint(10, 50)) for x in np.arange(4)]
        im = Image.open(file_path)
        wpercent = (basewidth/float(im.size[0]))
        hsize = int((float(im.size[1])*float(wpercent)))
        im = im.resize((basewidth,hsize), Image.ANTIALIAS)

        img, mask, tf_mask = gen_phrase_mask(im, phrase_list)

        filename = os.path.splitext(os.path.basename(file_path))[0]
        img_path = os.path.join(root + 'pics', filename + '_' + str(ind) + '.jpg')
        # mask_path = os.path.join(root + 'mask', filename + '_' + str(ind) + '.png')
        tf_mask_path = os.path.join(root + 'mask', filename + '_tf_' + str(ind) + '.png')
        # img.resize((640, 360)).save(img_path)

        arr = np.array(tf_mask)
        tf_mask_arr_2d = convert_to_tf_mask(arr)
        tf_mask = Image.fromarray(tf_mask_arr_2d)


        img.save(img_path, quality=60, optimize=True)
        # mask.save(mask_path)
        tf_mask.save(tf_mask_path)

        img.show()
        mask.show()
        # tf_mask.show()
        print(img_path, tf_mask_path)
        return img_path, tf_mask_path
    except Exception as e:
        print('exception!!!!!', str(e))
        return '', ''

def generate():
    # gen_pics_and_masks('./backs/dino.png', 0)
    path_to_backgrounds = '/mnt/hgfs/xubuntu64/wallpapers/'

    pool_func_args = []
    for dir, _, pics in os.walk(path_to_backgrounds):
        for pic in pics:
            file_path = os.path.join(dir, pic)
            for ind in range(5):
                pool_func_args.append((file_path, ind))

    pool = ThreadPool(4)
    import time
    start_time = time.time()
    results = pool.starmap(gen_pics_and_masks, pool_func_args)
    runtime = time.time() - start_time
    print(runtime)
    train_path = root + 'splits/train_tf.txt'
    with open(train_path, 'w') as f:
        for img_path, tf_mask_path in results:
            new_str = img_path + ' ' + tf_mask_path + '\n'
            f.write(new_str)

    pool.close()
    pool.join()

    phrase_list = [gen_random_phrase(random.randint(10, 50)) for x in
                   np.arange(4)]
    img, mask = gen_pics_and_masks('backs/horses.jpg', 0)
    print(np.random.randn(1, 2))
    print(len(words.words()))


if __name__ == "__main__":
    import time
    ts = time.time()
    generate()
    print(time.time() - ts)




