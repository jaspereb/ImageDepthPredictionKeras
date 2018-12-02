# Author: Jasper Brown
# jasperebrown@gmail.com
# 2019

import numpy as np
import cv2
import os

#Use to override image path
IMG_PATH = '/home/jasper/git/ImageDepthPredictionKeras/SampleData/depth1.png'

img = cv2.imread(IMG_PATH,0)
    
#img = np.asarray(img)
    
