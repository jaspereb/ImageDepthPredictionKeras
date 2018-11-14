from model import buildModel
from keras import backend as K
from dataGen import buildKittiList, KittiDataGenerator, testGenerator
import numpy as np

CPUMODE = False
if(CPUMODE):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
#Build dataset
rgbDataDir = "/home/jasper/Downloads/TestKittiDataset/RGBs"
depthDataDir = "/home/jasper/Downloads/TestKittiDataset/depths"
part, labels = buildKittiList(rgbDataDir, depthDataDir)

kittiGen = KittiDataGenerator(part['train'], labels)
kittiGenVal = KittiDataGenerator(part['validation'], labels)

#Build model
model = buildModel()

#Print model summary to file
with open('model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

#Fit model
model.fit_generator(kittiGen, validation_data=kittiGenVal, use_multiprocessing=True)


