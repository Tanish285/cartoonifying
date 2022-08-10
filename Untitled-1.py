from unittest import result
import pandas
import cv2
import numpy as np

def readimg(filename):
    img=cv2.imread(filename)
    return img

def egde_detection(img,line_width,blur):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayblur=cv2.medianBlur(gray,blur)
    edges=cv2.adaptiveThreshold(grayblur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_width,blur)
    return edges

def colorquant(img,k):
    data=np.float32(img).reshape((-1,3))
    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER,20,0.001)
    ret,label,center=cv2.kmeans(data,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    result=center[label.flatten()]
    result=result.reshape(img.shape)
    return result

img = readimg('E:\project\images.jpeg')
line_width=9
blur_value=7
totalcolors=4

edgeimg=egde_detection(img,line_width,blur_value)
img=colorquant(img,totalcolors)
blurred=cv2.bilateralFilter(img,d=7,sigmaColor=200,sigmaSpace=200)
cartoon=cv2.bitwise_and(blurred,blurred,mask=edgeimg)
cv2.imwrite('cartoon.jpg',cartoon)
