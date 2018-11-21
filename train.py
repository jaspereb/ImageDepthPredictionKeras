from model import buildModel
from keras import backend as K
#from dataGenerators import buildKittiList, KittiDataGenerator, testGenerator
#from dataGenerators import buildYCBList, YCBDataGenerator, testGenerator
from dataListers import buildRedwoodList
from dataGenerators import RedwoodDataGenerator, testGenerator
import numpy as np

CPUMODE = True
if(CPUMODE):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
#Build Redwood dataset
redwoodDir = "/media/jasper/107480BB7480A4D6/Users/Jasper/Datasets/Redwood"
part, labels = buildRedwoodList(redwoodDir)
RedGen = RedwoodDataGenerator(part['train'], labels)
RedGenVal = RedwoodDataGenerator(part['validation'], labels)

#Build model
model = buildModel()

#Print model summary to file
with open('model_summary.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

#Fit model
model.fit_generator(RedGen, validation_data=RedGenVal, use_multiprocessing=True)




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