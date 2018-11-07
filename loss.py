'''The loss function composed from a supervised loss (sum of L2 norm) and 
a regularisation term which penalises high depth gradients where there are 
not high image gradients '''
from keras import backend as K 
from keras.layers import Input
from keras.models import Model

import numpy as np


def supervisedLoss(yTrue, yPred):
    yTrue = K.flatten(yTrue)
    yPred = K.flatten(yPred)
    
    #Ignore pixels where depth info is not present or is inaccurate
#    zeros = np.where(yTrue == 0 or yTrue > 8000)
    
    error = yPred - yTrue
#    error[zeros] = 0

    #Using the L2 Norm
    norm = K.sum((K.abs(error))**2)
    norm = K.sqrt(norm)
    return norm
    
def smoothnessLoss(yTrue, yPred, Img):
    #Not really sure how do access the image data in keras, this could be very slow
    return K.sum(K.flatten(yTrue) - K.flatten(yPred)*K.flatten(Img))
#    return 0
#    RGB = Input(shape=(640,480,3,))
#    tempModel = Model(input=RGB, output)
#    print("ToDo")
#    return

def compositeLoss(inputs):
    yTrue = inputs[0] 
    yPred = inputs[1]
    Img = inputs[2]
    
    supervisedLossVal = supervisedLoss(yTrue, yPred)
    smoothnessLossVal = smoothnessLoss(yTrue, yPred, Img)
    compositeLoss = supervisedLossVal + smoothnessLossVal
    
    return compositeLoss