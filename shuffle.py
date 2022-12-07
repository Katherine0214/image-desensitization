import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('face.jpg')
M,N,_ = img.shape

r_ori = np.arange(M)
r_shuffle = r_ori.copy()
np.random.shuffle(r_shuffle)
r_restore = np.argsort(r_shuffle)   # 将r_shuffle中的元素从小到大排列，提取其在排列前对应的index(索引)输出

c_ori = np.arange(N)
c_shuffle = c_ori.copy()
np.random.shuffle(c_shuffle)
c_restore = np.argsort(c_shuffle)

img_r_emb = img.copy()[r_shuffle]
img_c_emb = img.copy()[:,c_shuffle]
img_emb = img_r_emb.copy()[:,c_shuffle]

img_restore = img_emb.copy()[:,c_restore]
img_restore = img_restore[r_restore]

cv2.imshow('ori',img)
cv2.imshow('encode',img_emb)
cv2.imshow('decode',img_restore)
cv2.waitKey(0)

