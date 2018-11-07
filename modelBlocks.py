from keras.layers import Conv2D, Activation, Add, BatchNormalization, Lambda
import tensorflow as tf

def type1Resblock(resInput, resStride, resFilter1, resFilter2):
    '''The type 1 residual block, input shape is (batch, height, width, resFilter1) and
    output is always the same shape'''
    
    #Top row first block
    top1 = Conv2D(filters=resFilter1, kernel_size=(1,1), strides=(resStride,resStride), padding='same', data_format='channels_last', use_bias=False)(resInput)
    top2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(top1)
    top3 = Activation('relu')(top2)
    
    #Top row second block
    top4 = Conv2D(filters=resFilter1, kernel_size=(3,3), strides=(resStride,resStride), padding='same', data_format='channels_last', use_bias=False)(top3)
    top5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(top4)
    top6 = Activation('relu')(top5)
    
    #Top row third block
    top7 = Conv2D(filters=resFilter2, kernel_size=(1,1), strides=(resStride,resStride), padding='same', data_format='channels_last', use_bias=False)(top6)
    top8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(top7)
    
    outSum = Add()([top8, resInput])
    output = Activation('relu')(outSum)
    
    return output

def type2Resblock(resInput, resStride, resFilter1, resFilter2):
    '''The type 1 residual block, input shape is (batch, height, width, resFilter1) and
    output is (batch, height/stride, width/stride, resFilter2)'''
    #Top row first block
    top1 = Conv2D(filters=resFilter1, kernel_size=(1,1), strides=(resStride,resStride), padding='same', data_format='channels_last', use_bias=False)(resInput)
    top2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(top1)
    top3 = Activation('relu')(top2)
    
    #Top row second block
    top4 = Conv2D(filters=resFilter1, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last', use_bias=False)(top3)
    top5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(top4)
    top6 = Activation('relu')(top5)
    
    #Top row third block
    top7 = Conv2D(filters=resFilter2, kernel_size=(1,1), strides=(1,1), padding='same', data_format='channels_last', use_bias=False)(top6)
    top8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(top7)
    
    #Bottom row
    bottom1 = Conv2D(filters=resFilter2, kernel_size=(1,1), strides=(resStride,resStride), padding='same', data_format='channels_last', use_bias=False)(resInput)
    bottom2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00000001)(bottom1)
    
    outSum = Add()([top8, bottom2])
    output = Activation('relu')(outSum)
    
    return output
    
def interleave(convs):
    # Concatenate convolution outputs
    sh = convs[0].get_shape().as_list()
    dim = len(sh[1:-1])
    tmp1 = tf.reshape(convs[0], [-1] + sh[-dim:])
    tmp2 = tf.reshape(convs[1], [-1] + sh[-dim:])
    tmp3 = tf.reshape(convs[2], [-1] + sh[-dim:])
    tmp4 = tf.reshape(convs[3], [-1] + sh[-dim:])
    
    # horizontal concat
    concat1 = tf.concat([tmp1, tmp3], 2)
    concat2 = tf.concat([tmp2, tmp4], 2)
    # vertical concat
    concat_final = tf.concat([concat1, concat2], 1)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    interleaved = tf.reshape(concat_final, out_size)
    return interleaved
    
def concatPad(inputs):
    #Concatenates the features of same scale having different height / width due to striding / upsampling
    upProjection = inputs[0]
    resBlock = inputs[1]
    
    shape1 = resBlock.get_shape().as_list()[1:-1]
    shape2 = upProjection.get_shape().as_list()[1:-1]
    padding = [a_i - b_i for a_i, b_i in zip(shape2, shape1)]
    blockPadded = tf.pad(resBlock, [[0, 0], [0, padding[0]], [0, padding[1]], [0, 0]])
    concatenated = tf.concat([upProjection, blockPadded], 3)
    return concatenated    
    
def upProjectFast(projInput, numChannels):
    ''' The fast upProject module from Laina et al 2016. Some tensorflow code 
    for reshaping comes from Yevkuzn. 
    
    Input is (batch, height, width, filters) and output shape 
    is (batch, 2*height, 2*width, filters/2)'''
    #Top Half
    conv1 = Conv2D(filters=numChannels, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    conv2 = Conv2D(filters=numChannels, kernel_size=(2,3), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    conv3 = Conv2D(filters=numChannels, kernel_size=(3,2), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    conv4 = Conv2D(filters=numChannels, kernel_size=(2,2), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    
    #Bottom Half
    conv5 = Conv2D(filters=numChannels, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    conv6 = Conv2D(filters=numChannels, kernel_size=(2,3), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    conv7 = Conv2D(filters=numChannels, kernel_size=(3,2), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    conv8 = Conv2D(filters=numChannels, kernel_size=(2,2), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(projInput)
    
    #Interleave them
    topInterleaved = Lambda(interleave)([conv1,conv2,conv3,conv4])
    bottomInterleaved = Lambda(interleave)([conv5,conv6,conv7,conv8])
    
    #Apply the extra convolution on the top path
    top1 = Activation('relu')(topInterleaved)
    top2 = Conv2D(filters=numChannels, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last', use_bias=True)(top1)
    
    #Combine the outputs
    convSum = Add()([top2, bottomInterleaved])
    output = Activation('relu')(convSum)
    
    return output
    
    
    