''' Builds a keras model for the supervised version of 
Semi-supervised Deep Learning for Monocular Depth Map Prediction'''

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
#from loss import compositeLoss
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import numpy as np
import tensorflow as tf

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
    norm = K.abs(error)
    loss = K.sum(K.square(norm))
#    loss = K.sqrt(norm)
    return loss

def buildModelCAE():
    # ==========  Define model  ==========
    RGB = Input(shape=(640,480,3,))
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(RGB)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    #Compile the full model
    model = Model(inputs=RGB, outputs=decoded)
    model.compile(optimizer='adadelta', loss='mean_squared_error')
    model.summary()
    
    return model


if __name__ == "__main__":
    model = buildModelCAE()
    
    
    
    
    
    
