from itertools import chain
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode

font = TTFont(sys.argv[1])

for cmap in font['cmap'].tables:
    if cmap.isUnicode():
        if ord('的') in cmap.cmap:
            print('Exist')
    
print('end')
