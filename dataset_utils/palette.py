from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from alphabet_ru_short import alphabet, ALPHABET_DICT_COLORED
source_img = Image.new('RGBA', (600, 300),
                        (255, 255, 255, 255))

draw = ImageDraw.Draw(source_img)

rect_size = 30
col_len = 10

for ind, ch in enumerate(alphabet):
    shift = int(ind/col_len) * 120
    ch_color = ALPHABET_DICT_COLORED[ch] + (255,)
    if ch == '/':
        ch = 'фон'
    draw.rectangle(((0 + shift, (ind % col_len)*rect_size), (50 + shift, (ind % col_len)*rect_size + rect_size)), fill=ch_color)
    draw.text((shift + rect_size*2, (ind % col_len)*rect_size), '- ' + ch, font=ImageFont.truetype("Arial.ttf", 20),
              fill=(0, 0, 0, 255))
source_img.show()
source_img.save('palette.png')