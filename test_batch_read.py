from font_reader import FontReader
from PIL import Image, ImageFont, ImageDraw
import time

batch_size = 100
col_size = 10

img_size = 48

std_reader = FontReader('data/standard/mingliu.TTF', img_size-12, img_size)
sty_reader = FontReader('data/train/NotoSansCJKtc-Light.otf', img_size-12, img_size)

std_batch = Image.new("L", (img_size*batch_size//col_size, img_size*col_size), color=(0))
sty_batch = Image.new("L", (img_size*batch_size//col_size, img_size*col_size), color=(0))

with open('character_set/character_set_1798', 'r') as char_set:
    char_batch = char_set.readlines()
    char_batch = [x.strip() for x in char_batch] 

    start = time.time()
    for i in range(batch_size):
        c = char_batch[i]
        idx = i // col_size
        idy = i % col_size

        std_img = std_reader.get_image(c)
        std_batch.paste(std_img, (img_size*idx, img_size*idy))

        sty_img = sty_reader.get_image(c)
        sty_batch.paste(sty_img, (img_size*idx, img_size*idy))

    print(time.time() - start)

std_batch.show()
sty_batch.show()
