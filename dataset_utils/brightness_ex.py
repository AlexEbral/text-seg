from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from alphabet_ru_short import alphabet, ALPHABET_DICT_COLORED
source_img = Image.new('RGBA', (600, 200), (0, 0, 0, 255))

draw = ImageDraw.Draw(source_img)

draw.text((10, 60), 'Пример', font=ImageFont.truetype("Arial.ttf", 45), fill=(5, 5, 5, 255))
draw.text((200, 60), 'Пример', font=ImageFont.truetype("Arial.ttf", 45), fill=(100, 100, 100, 255))
draw.text((400, 60), 'Пример', font=ImageFont.truetype("Arial.ttf", 45), fill=(200, 200, 200, 255))
source_img.show()
source_img.save('brg.png')