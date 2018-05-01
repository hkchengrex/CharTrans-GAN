from PIL import Image, ImageFont, ImageDraw
import numpy as np

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode

from itertools import chain

class FontReader:

    def __init__(self, font_path, font_size, image_size):
        self.font = ImageFont.truetype(font_path, font_size*2)
        self.font_size = font_size
        self.image_size = image_size

    def get_image(self, character):
        """
        Returns PIL image
        """
        # First draw once to find the center of mass
        cg_image = Image.new("L", (self.image_size*4, self.image_size*4), color=(0))
        cg_draw = ImageDraw.Draw(cg_image)
        cg_draw.text((0, 0), character, (255), font=self.font)

        try:
            np_image = np.array(cg_image)
            non_empty_col = np.nonzero(np_image.any(axis=0))[0]
            non_empty_row = np.nonzero(np_image.any(axis=1))[0]
            start_x = non_empty_col[0] - 1
            last_x = non_empty_col[-1] + 1
            start_y = non_empty_row[0] - 1 
            last_y = non_empty_row[-1] + 1
        except IndexError:
            return Image.new("L", (self.image_size, self.image_size), color=(0))

        cropped = cg_image.crop((start_x, start_y, last_x, last_y))

        return cropped.resize((self.image_size, self.image_size))
