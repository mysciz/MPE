from  PIL import Image
import numpy as np
from pylab import *
from numpy import *
from numpy import random
from scipy.ndimage import filters
import rof
 
#添加中文字体
from matplotlib.font_manager import FontProperties
 
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
 
figure()
gray()
#使用噪声创建合成图像
im = zeros((500,500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30*random.standard_normal((500,500))
subplot(1,3,1)
imshow(im)
axis("off")
title(u'原图', fontproperties=font)
 
U,T = rof.denoise(im,im)
G = filters.gaussian_filter(im,10)
subplot(1,3,2)
imshow(U)
axis("off")
title(u'rof去噪图像', fontproperties=font)
 
subplot(1,3,3)
imshow(G)
axis("off")
title(u'高斯去噪图像', fontproperties=font)
 
show()
#保存生成结果
#from imageio import imsave
#imsave('synth_rof.pdf',U)
#imsave('synth_gaussian.pdf',G)