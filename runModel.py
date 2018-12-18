# Author: Jasper Brown
# jasperebrown@gmail.com
# 2019

import numpy as np
from keras.models import load_model
import tensorflow as tf
from model import lossFn
from PIL import Image
import matplotlib.pyplot as plt

print("Loading model...")
model = load_model('/mnt/0FEF1F423FF4C54B/TrainedModels/AED/Thrs/weights.22-574127723349.85.hdf5', 
                          custom_objects={'tf':tf, 'lossFn':lossFn})

#single mode
dim = (640,480)
n_channels = 3
#dimOut = (320,240)
dimOut = dim

#Load prediction data
X = np.empty((1, dim[0], dim[1], n_channels))
y = np.empty((dimOut[0], dimOut[1]))

IMGPATH = '/home/jasper/git/ImageDepthPredictionKeras/SampleData/rgb4.jpg'

X[0,] = np.swapaxes(np.array(Image.open(IMGPATH)),0,1)
y = model.predict(X, 1, verbose=1)
y = y[0,:,:,0]
y = np.swapaxes(y,0,1)

plt.imshow(y) 
plt.show()













##   == Batch mode ==
#batch_size = 2
#dim = (640,480)
#n_channels = 3
#dimOut = (320,240)
#
##Load prediction data
#X = np.empty((batch_size, dim[0], dim[1], n_channels))
#y = np.empty((batch_size, dimOut[0], dimOut[1]))
#
#IMGPATH = ['/home/jasper/git/ImageDepthPredictionKeras/SampleData/rgb1.jpg',
#           '/home/jasper/git/ImageDepthPredictionKeras/SampleData/rgb2.jpg']
#
#for i, ID in enumerate(IMGPATH):
#    X[i,] = np.swapaxes(np.array(Image.open(IMGPATH)),0,1)
    
#Predict
#y = model.predict(X, batch_size, verbose=1)

## Store ground truth
#img = Image.open(self.labels[ID])
#img.thumbnail((self.dimOut), Image.ANTIALIAS)
#
#y[i,] = np.swapaxes(np.array(img),0,1)



#Save output