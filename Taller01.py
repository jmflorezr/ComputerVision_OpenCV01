import cv2 as cv
import numpy as np
import sys
import os
from matplotlib import pyplot as plt

#Importa la Imagen desde la carpeta de trabajo
basicColor = cv.imread('/content/ImagenJPG.jpg')
#Convierte RGB a BGR
gray = cv.cvtColor(basicColor,cv.COLOR_BGR2GRAY)
#Propiedades de la foto
histograma = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(histograma, color='gray')
plt.xlim([0, 256])
plt.show()
plt.imshow(gray)
plt.title('Imagen grises')
plt.show()

#Tomando la imagen de grises BGR2GRAY, se extraen los pixeles blancos y negros ==> Sumatoria
number_of_white_pix = np.sum(gray == 255)
number_of_black_pix = np.sum(gray == 0)
Total_Pixeles =  number_of_white_pix + number_of_black_pix
print('Total pixeles Imagen:', Total_Pixeles)

#Canales de la imagen y su histograma por color
bgr_planes = cv.split(basicColor)
histSize = 256
histRange = (0, 256) # Limite Superior
accumulate = False
b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)
hist_w = 512
hist_h = 400
bin_w = int(round( hist_w/histSize ))
histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
for i in range(1, histSize):
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(b_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(b_hist[i]) ),
            ( 255, 0, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(g_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(g_hist[i]) ),
            ( 0, 255, 0), thickness=2)
    cv.line(histImage, ( bin_w*(i-1), hist_h - int(r_hist[i-1]) ),
            ( bin_w*(i), hist_h - int(r_hist[i]) ),
            ( 0, 0, 255), thickness=2)
plt.imshow(histImage)
plt.title('Histograma RGB de los canales')
plt.show()
print(bgr_planes)

#Version Binaria
ret, Ibw_otsu = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
plt.imshow(Ibw_otsu)
plt.title('Imagen Binaria')
plt.show()
