import os

fonts_dir = ''

def fonts_list(russian=False):
    fonts_list = []
    bad_fonts_list = []

    if russian:
        fonts_dir = ''
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
    fonts_list = fonts_list()
    print(fonts_list[0])
    print(fonts_list[10])
    print(len(fonts_list))

    for font in fonts_list:
        if font[-3:].lower() != 'ttf':
            print(font)