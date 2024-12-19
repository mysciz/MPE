from PIL import Image
from pylab import *
import rof
from scipy.ndimage import filters
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=10)
figure()
gray()
 
im = array(Image.open('Images\image copy.png').convert('L'))
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
title(u'高斯去噪图像', fontproperties=font)
axis("off")
show()