# from get_fonts_list import fonts_list
from PIL import Image, ImageDraw, ImageFont
from getkey import getkey, keys
import os

base_dir = './'
fonts_dir = ''

def fonts_list():
    fonts_list = []
    bad_fonts_list = []

    for dir, _, fonts in os.walk(fonts_dir):
        for font in fonts:
            if font[-3:] != 'ttf':
                continue
            if '_bad_fnt' in font:
                bad_fonts_list.append(os.path.join(dir, font))
            else:
                fonts_list.append(os.path.join(dir, font))
    return fonts_list, bad_fonts_list


if __name__ == "__main__":
    fonts_list, bad_fonts_list  = fonts_list()
    fonts_list_len = len(fonts_list)
    bad_fonts_list_len = len(bad_fonts_list)
    print('fonts num: ', fonts_list_len)
    print('bad fonts num: ', bad_fonts_list_len)
    ind = 0
    txt = Image.new('RGBA', (300, 1200),
                    (255, 255, 255, 255))
    d = ImageDraw.Draw(txt)
    d.fontmode = "1"
    h = 30

    fnts_dict = {}
    while True:
        text_fnt = fonts_list[ind]
        print('#', ind + 1, ' font:', text_fnt)
        fnts_dict[str(ind + 1)] = text_fnt
        fnt = ImageFont.truetype(text_fnt, h)
        d.text((0, h*(ind % h)), str(ind + 1) + 'Русский123!;.', font=fnt, fill=(0, 0, 0))
        ind += 1

        if ind % h == 0 or ind >= len(fonts_list):
            txt.show()
            # getkey()
            pics_inds = input().split()

            print(pics_inds)
            for indx in pics_inds:
                if '-' in indx:
                    ind_range = indx.split('-')
                    for i in range(int(ind_range[0]), int(ind_range[1]) + 1):
                        src = fnts_dict[str(i)].split('.')
                        dst = src[0] + '_bad_fnt.' + src[1]
                        os.rename(fnts_dict[str(i)], dst)
                else:
                    src = fnts_dict[indx].split('.')
                    dst = src[0] + '_bad_fnt.' + src[1]
                    os.rename(fnts_dict[indx], dst)

            txt = Image.new('RGBA', (300, 1200),
                            (255, 255, 255, 255))
            d = ImageDraw.Draw(txt)
            d.fontmode = "1"

            if ind >= len(fonts_list):
                break


