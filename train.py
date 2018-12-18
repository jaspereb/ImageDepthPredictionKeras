from model import buildModelBaseline, lossFn
import tensorflow as tf
from keras import backend as K
from keras import models
from keras.callbacks import ModelCheckpoint, TensorBoard
#from dataGenerators import buildKittiList, KittiDataGenerator, testGenerator
#from dataGenerators import buildYCBList, YCBDataGenerator, testGenerator
from dataListers import buildRedwoodList
from dataGenerators import RedwoodDataGenerator, testGenerator
import numpy as np
import pickle
from time import gmtime, strftime
import sys

CPUMODE = False
LOADDATASET = True

batch_size = 2
if(CPUMODE):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    

if(LOADDATASET):
    print("Loading dataset")
    with open('./part', 'rb') as file:
            part = pickle.load(file)
    
    with open('./labels', 'rb') as file:
            labels = pickle.load(file)
else:
    #Build Redwood dataset
    print("Building dataset")
    redwoodDir = "/mnt/0FEF1F423FF4C54B/Datasets/Redwood"
    part, labels = buildRedwoodList(redwoodDir, 0.998)
    
    with open('./part', 'wb') as file:
            pickle.dump(part, file)
    
    with open('./labels', 'wb') as file:
            pickle.dump(labels, file)

RedGen = RedwoodDataGenerator(part['train'], labels, batch_size=batch_size)
RedGenVal = RedwoodDataGenerator(part['validation'], labels, batch_size=batch_size)

print("Building model...")
model = buildModelBaseline()

#print("Loading model...")
#model = models.load_model('/mnt/0FEF1F423FF4C54B/TrainedModels/weights.05-81862692952.85.hdf5', 
#                          custom_objects={'tf':tf, 'lossFn':lossFn})

checkpointer = ModelCheckpoint(filepath='/mnt/0FEF1F423FF4C54B/TrainedModels/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False)

boarder = TensorBoard(log_dir='./logs')


#Print model summary to file
with open('model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

#Fit model
steps = (len(part['train'])/batch_size)
steps = 5000

history = model.fit_generator(RedGen, epochs=50, steps_per_epoch=steps, validation_data=RedGenVal, 
                              use_multiprocessing=True, callbacks=[checkpointer, boarder])

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
