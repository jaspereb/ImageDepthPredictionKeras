from model import buildModel, lossFn
import tensorflow as tf
from keras import backend as K
from keras import models
from keras.callbacks import ModelCheckpoint
#from dataGenerators import buildKittiList, KittiDataGenerator, testGenerator
#from dataGenerators import buildYCBList, YCBDataGenerator, testGenerator
from dataListers import buildRedwoodList
from dataGenerators import RedwoodDataGenerator, testGenerator
import numpy as np
import pickle
from time import gmtime, strftime
import sys

CPUMODE = False
if(CPUMODE):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
#Build Redwood dataset
#print("Building dataset")
#redwoodDir = "/mnt/0FEF1F423FF4C54B/Datasets/Redwood"
#part, labels = buildRedwoodList(redwoodDir)
#
#with open('./part', 'wb') as file:
#        pickle.dump(part, file)
#
#with open('./labels', 'wb') as file:
#        pickle.dump(labels, file)

print("Loading dataset")
with open('./part', 'rb') as file:
        part = pickle.load(file)

with open('./labels', 'rb') as file:
        labels = pickle.load(file)

RedGen = RedwoodDataGenerator(part['train'], labels)
RedGenVal = RedwoodDataGenerator(part['validation'], labels)

#print("Building model...")
#model = buildModel()

print("Loading model...")
model = models.load_model('/home/jasper/git/ImageDepthPredictionKeras/weights.02-246543.44.hdf5', 
                          custom_objects={'tf':tf, 'lossFn':lossFn})

checkpointer = ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)

#Print model summary to file
with open('model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

#Fit model
history = model.fit_generator(RedGen, epochs=10, validation_data=RedGenVal, use_multiprocessing=True, callbacks=[checkpointer])

with open('./trainingHistory', 'wb') as hist:
        pickle.dump(history.history, hist)

modelPath = './savedModels/' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.h5'
model.save(modelPath)

#OTHER DATASETS
##Build kitti dataset
#rgbDataDir = "/home/jasper/Downloads/TestKittiDataset/RGBs"
#depthDataDir = "/home/jasper/Downloads/TestKittiDataset/depths"
#part, labels = buildKittiList(rgbDataDir, depthDataDir)
#
#kittiGen = KittiDataGenerator(part['train'], labels)
#kittiGenVal = KittiDataGenerator(part['validation'], labels)

##Build YCB dataset
#YCBDataDir = "/media/jasper/107480BB7480A4D6/Users/Jasper/Datasets/YCB"
#part, labels = buildYCBList(YCBDataDir)
#
#YCBGen = YCBDataGenerator(part['train'], labels)
#YCBGenVal = YCBDataGenerator(part['validation'], labels)
