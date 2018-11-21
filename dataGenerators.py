'''Create a datagenerator for RGB/Depth image pairs. Adapted from:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''

import numpy as np
import keras
import h5py
from PIL import Image
from dataListers import *

# ========================================== THE DATA GENERATORS ==========================================
#For the redwood depth map dataset
class RedwoodDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=(640,480), dimOut=(320,240), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dimOut = dimOut
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dimOut[0], self.dimOut[1]))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.swapaxes(np.array(Image.open(ID)),0,1)
            

            # Store ground truth
            img = Image.open(self.labels[ID])
            img.thumbnail((self.dimOut), Image.ANTIALIAS)
            
            y[i,] = np.swapaxes(np.array(img),0,1)
            
        return X, y
    
#For the YCB depth map dataset
class YCBDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=(640,480), dimOut=(320,240), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dimOut = dimOut
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dimOut[0], self.dimOut[1]))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.swapaxes(np.array(Image.open(ID)),0,1)
            

            # Store ground truth from h5 files
            img = h5py.File(self.labels[ID],'r')
            img = img['depth'].value
            
            #Downscale data, this does not use antialiasing
            img = img[::2,::2]
            
            y[i,] = np.swapaxes(img,0,1)
            
        return X, y
    
#For the kitti depth map dataset
class KittiDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=(640,480), dimOut=(320,240), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dimOut = dimOut
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dimOut[0], self.dimOut[1]))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.swapaxes(np.array(Image.open(ID)),0,1)
            

            # Store ground truth
            img = Image.open(self.labels[ID])
            img.thumbnail((self.dimOut), Image.ANTIALIAS)
            
            y[i,] = np.swapaxes(np.array(img),0,1)
            
        return X, y
    
#For the washington RGBD Objects Dataset
class RGBDDataGenerator(keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=4, dim=(640,480), dimOut=(320,240), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.dimOut = dimOut
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.empty((self.batch_size, self.dimOut[0], self.dimOut[1]))
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.swapaxes(np.array(Image.open(ID)),0,1)
            

            # Store ground truth
            img = Image.open(self.labels[ID])
            img.thumbnail((self.dimOut), Image.ANTIALIAS)
            
            y[i,] = np.swapaxes(np.array(img),0,1)
            
        return X, y
    
class testGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=1, dim=(640,480), n_channels=3):
        print("RUNNING INIT")
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        
    def __len__(self):
        print("Running LEN")
        'Denotes the number of batches per epoch'
        return self.batch_size

    def __getitem__(self, index):
        print("GETTING ITEM")
        batch_features = np.random.rand(self.batch_size, 640,480, 3)
        batch_labels = np.random.rand(self.batch_size, 320,240, 1)
    
        print("SHAPE IS")
        print(batch_features.shape)
        return batch_features,batch_labels
    
    
if __name__ == "__main__":
#    rgbDataDir = "/home/jasper/Downloads/TestKittiDataset/RGBs"
#    depthDataDir = "/home/jasper/Downloads/TestKittiDataset/depths"
#    part, labels = buildKittiList(rgbDataDir, depthDataDir)
    
#    ycbDir = "/media/jasper/107480BB7480A4D6/Users/Jasper/Datasets/YCB"
#    part, labels = buildYCBList(ycbDir)
#    YCBGen = YCBDataGenerator(part['train'], labels)
    
    redwoodDir = "/media/jasper/107480BB7480A4D6/Users/Jasper/Datasets/Redwood"
    part, labels = buildRedwoodList(redwoodDir)
    RedGen = RedwoodDataGenerator(part['train'], labels)
    
#    kittiGen = KittiDataGenerator(part['train'], labels)
    
#    rgbDataDir = "/home/jasper/Downloads/rgbd-dataset/apple/apple_1/images"
#    depthDataDir = "/home/jasper/Downloads/rgbd-dataset/apple/apple_1/depth"
#    part, labels = buildWashingtonList(rgbDataDir, depthDataDir, 0.2)
#    
#    washingtonGen = RGBDDataGenerator(part['train'], labels)