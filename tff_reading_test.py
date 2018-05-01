import sys
import os

from PIL import Image, ImageFont, ImageDraw
from matplotlib.pyplot import imshow
import numpy as np
from scipy.ndimage.measurements import center_of_mass

image = Image.new("L", (80, 80), color=(1))
draw = ImageDraw.Draw(image)

path = "data/train/wt004.ttf"
#path = "data/train/NotoSansCJKtc-Light.otf"
font = ImageFont.truetype(path, 64)

#w, h = draw.textsize("的", font=font)
#draw.text((40-w/2, 40-h/2), "的", font=font)

draw.text((0, 10), "的", (255),font=font)
print(center_of_mass(np.array(image)))

image.show()
