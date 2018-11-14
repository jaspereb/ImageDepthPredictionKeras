''' Builds a keras model for the supervised version of 
Semi-supervised Deep Learning for Monocular Depth Map Prediction'''

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from modelBlocks import type1Resblock, type2Resblock, upProjectFast, concatPad
from loss import compositeLoss
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Activation, Lambda, Dropout
import numpy as np

def lossFn(y_true,y_pred):
    #y_true is a (4,) tensor
    yTrue = K.tf.reshape(y_true, [-1])
    yPred = K.tf.reshape(y_pred, [-1])
    
    #Ignore pixels where depth info is not present or is inaccurate
    nonZero = (yTrue > 0)
    inRange = yTrue < 8000
    valids = K.tf.logical_and(nonZero, inRange)
    
    error = yPred - yTrue
    error = K.tf.boolean_mask(error, valids)

    #Using the L2 Norm
    norm = K.sum((K.abs(error))**2)
    norm = K.sqrt(norm)
    return norm

def buildModel():
    # ==========  Define model  ==========
    RGB = Input(shape=(640,480,3,))
    
    #First convolution
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', data_format='channels_last', use_bias=False)(RGB)
    bn1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(conv1)
    rel1 = Activation('relu')(bn1)
    
    #Max pooling
    mp1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', data_format='channels_last')(rel1)
    
    #Resblocks (input, stride, filtersIn, filtersOut)
    res1 = type2Resblock(mp1, 1, 64, 256)
    res2 = type1Resblock(res1, 1, 256, 256)
    res3 = type1Resblock(res2, 1, 256, 256)
    res4 = type2Resblock(res3, 2, 256, 512)
    
    res5 = type1Resblock(res4, 1, 512, 512)
    res6 = type1Resblock(res5, 1, 512, 512)
    res7 = type1Resblock(res6, 1, 512, 512)
    res8 = type2Resblock(res7, 2, 512, 1024)
    
    res9 = type1Resblock(res8, 1, 1024, 1024)
    res10 = type1Resblock(res9, 1, 1024, 1024)
    res11 = type1Resblock(res10, 1, 1024, 1024)
    res12 = type1Resblock(res11, 1, 1024, 1024)
    res13 = type1Resblock(res12, 1, 1024, 1024)
    res14 = type2Resblock(res13, 2, 1024, 2048)
    
    res15 = type1Resblock(res14, 1, 2048, 2048)
    res16 = type1Resblock(res15, 1, 2048, 2048)
    conv2 = Conv2D(filters=1024, kernel_size=(1,1), strides=(1,1), padding='same', data_format='channels_last', use_bias=False)(res16)
    
    #Up projection blocks (input, numChannelsOut)
    up1 = upProjectFast(conv2, 512)
    merge1 = Lambda(concatPad)([up1, res13])
    
    up2 = upProjectFast(merge1, 256)
    merge2 = Lambda(concatPad)([up2, res7])
    
    up3 = upProjectFast(merge2, 128)
    merge3 = Lambda(concatPad)([up3, res3])
    
    up4 = upProjectFast(merge3, 64)
    drop1 = Dropout(0.5)(up4)
    
    #Final convolution
    DEPTH = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last', use_bias=False)(drop1)
    
    

#    #ToDo Change loss to use the smoothness component as well
#    #Loss Calculation
#    LABELS = Input(name='lossLabels', shape=(640,480,3,))
#    
#    #To allow the smoothness loss we use a lambda function as in
#    # https://github.com/keras-team/keras/blob/master/examples/image_ocr.py
#    LOSS = Lambda(compositeLoss, name='lossLambda')([DEPTH, LABELS, RGB])
#    model.compile(optimizer, loss={'lossLambda': lambda y_true, y_pred: y_pred})
    
    
    #Compile the full model
    model = Model(inputs=RGB, outputs=DEPTH)
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer, loss=lossFn)
#    model.summary()
    
    return model
    
if __name__ == "__main__":
    model = buildModel()
    
    
    
    
    
    