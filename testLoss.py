# Author: Jasper Brown
# jasperebrown@gmail.com
# 2019

#Test the output of the loss function by running it against a ground truth output

from model import lossFn
import numpy as np
from PIL import Image
from keras import backend as K
#import tensorflow as tf

IMGPATH = '/home/jasper/git/ImageDepthPredictionKeras/SampleData/sequential/depth1.png'
img = Image.open(IMGPATH)
img.thumbnail((320,240), Image.ANTIALIAS)

yIn1 = np.swapaxes(np.array(img),0,1)

IMGPATH = '/home/jasper/git/ImageDepthPredictionKeras/SampleData/sequential/depth2.png'
img = Image.open(IMGPATH)
img.thumbnail((320,240), Image.ANTIALIAS)

yIn2 = np.swapaxes(np.array(img),0,1)
            


#y_true is a (4,) tensor
yTrue = K.tf.reshape(yIn1, [-1])
yPred = K.tf.reshape(yIn2, [-1])

#Ignore pixels where depth info is not present or is inaccurate
nonZero = (yTrue > 0)
inRange = yTrue < 8000
valids = K.tf.logical_and(nonZero, inRange)

error = yPred - yTrue
error = K.tf.boolean_mask(error, valids)

#Using the L2 Norm
norm = K.abs(error)
loss = K.sum(K.square(norm))

print(norm)

# Launch the graph in a session.
sess = K.tf.Session()

# Evaluate the tensor `c`.
print(sess.run(loss))