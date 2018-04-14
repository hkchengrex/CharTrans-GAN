import sys
import os

from PIL import Image, ImageFont, ImageDraw
from matplotlib.pyplot import imshow

image = Image.new("RGB", (128, 128), color=(1,1,1))
draw = ImageDraw.Draw(image)

path = "data/train/SentyTEA.ttf"

font = ImageFont.truetype(path, 64)
draw.text((0, 0), "的", font=font)

font = ImageFont.truetype(path, 64)
draw.text((64, 0), "而", font=font)

font = ImageFont.truetype(path, 64)
draw.text((0, 64), "繁", font=font)

font = ImageFont.truetype(path, 64)
draw.text((64, 64), "龍", font=font)

image.show()
