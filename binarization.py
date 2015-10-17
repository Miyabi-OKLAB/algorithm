#! /usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
from matplotlib import pylab as plt

# 出力テキストデータ
FILENAME = 'test.txt'

img = np.array(Image.open('YU01.bmp').convert('L'))
"""
plt.imshow(img)
plt.show()
"""
# 1/10にする（300x300 -> 30x30）
img = (img[::10, ::10] + img[1::10, ::10] + img[::10, 1::10] + img[1::10, 1::10])/4

f = open(FILENAME, "w")

for i in img:
	for item in i:
		if(item > 50): 
			f.write("0")
		else:
			f.write("1")
		f.write(",")
	# スカラーにするときは↓をコメントアウト
	f.write("\n")

