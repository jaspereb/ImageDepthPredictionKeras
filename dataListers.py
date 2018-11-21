'''Create a datagenerator for RGB/Depth image pairs. Adapted from:
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
'''

import numpy as np
import glob
import os
import math
import h5py
from PIL import Image


# ========================================== THE LIST GENERATORS ==========================================
def buildRedwoodList(RedwoodDir, trainFrac=0.8):
    '''Builds a list of all the RGB/Depth image pairs from the redwood dataset.
    The Redwood Directory should have all of the individual scans folders in it, eg
    05507, 05628 etc. The images are jpg and the depth maps are png. Use the python script
    in the Dataset Bash Scripts folder to dl it and the bash command to unzip it'''
    
    #Find all the jpg files two directories down from the dataset folder
    path = os.path.join(RedwoodDir, '*', '*', '*.jpg')
    rgbFiles = glob.glob(path)
    path = os.path.join(RedwoodDir, '*', '*', '*.png')
    depthFilesAvail = glob.glob(path)
    depthTimes = np.zeros(len(depthFilesAvail))
    depthFiles = []
      
    print("A total of {} RGB files were found".format(len(rgbFiles)))
    print("Attempting to time match these against {} depth images".format(len(depthFilesAvail)))

    #Generate list of depth times to match against
    for depthIndex in range(0,len(depthFilesAvail)):
        _,depthTime = os.path.split(depthFilesAvail[depthIndex])
        depthTime,_ = os.path.splitext(depthTime)
        #Split at hyphen to get microseconds
        depthTime = int(depthTime.split('-')[1])
        depthTimes[depthIndex] = depthTime
    
    #Match each RGB image to a depth image
    for rgbFileCurrent in rgbFiles:
        _,rgbTime = os.path.split(rgbFileCurrent)
        rgbTime,_ = os.path.splitext(rgbTime)
        #Split at hyphen
        rgbTime = int(rgbTime.split('-')[1])
        
        depthID = (np.abs(depthTimes - rgbTime)).argmin()
        
        #Check the time difference is < 0.1sec
        if(abs(rgbTime - depthTimes[depthID]) > 100000):
            print("Skipping image, no matching depth frame within 0.1 seconds of timestamp:")
            print(rgbFileCurrent)
            rgbFiles.remove(rgbFileCurrent)
            continue
        
        if(os.path.isfile(depthFilesAvail[depthID])):
            depthFiles.append(depthFilesAvail[depthID])
        else:
            rgbFiles.remove(rgbFileCurrent)
            print("Warning: A matching depth image could not be found for the file:")
            print(rgbFileCurrent)
            continue
    
    if(len(depthFiles) < 1):
        print("\n\n=================== ERROR: No RGB/Depth pairs found ===================")
        print("Check the file naming convention matches that specified in the dataGen.py functions")
        return
    
    #Split into training and validation. Test should be split manually.
    numTrain = int(math.floor(trainFrac*len(rgbFiles)))
    numVal = len(rgbFiles)-numTrain
    
    #Make the dictionaries
    labels = {}
    partition = {}
    
    index = 0
    for index in range(0,len(rgbFiles)):
        labels[rgbFiles[index]] = depthFiles[index]
        
    partition['train'] = rgbFiles[:numTrain]
    partition['validation'] = rgbFiles[-numVal:]
    
    print("A total of {} RGB/Depth pairs will be used for training and {} for validation".format(numTrain, numVal))
    
    #Partition now looks like
    # {'train': ['file1', 'file2', 'file3'], 'validation': ['file4']}
    
    return partition, labels

def buildYCBList(YCBDir, trainFrac=0.8):
    '''Builds a list of all the RGB/Depth image pairs from the YCB dataset.
    The YCB Directory should have all of the individual object folders in it, eg
    001_Chips_can, 002_master_chef_can etc. All .h5 files are added to the datalist
    except for the calibration.h5 one'''
    
    #Find all the jpg files one directory down from the dataset folder
    path = os.path.join(YCBDir, '*', '*.jpg')
    rgbFiles = glob.glob(path)
    depthFiles = []
    
    print("A total of {} RGB files were found".format(len(rgbFiles)))

    #Match each RGB image to a depth image
    for rgbFileCurrent in rgbFiles:
        depthFile,_ = os.path.splitext(rgbFileCurrent)
        depthFile = depthFile + '.h5'
        
        if(os.path.isfile(depthFile)):
            depthFiles.append(depthFile)
        else:
            rgbFiles.remove(rgbFileCurrent)
            print("Warning: A matching depth image could not be found for the file:")
            print(rgbFileCurrent)
    
    if(len(depthFiles) < 1):
        print("\n\n=================== ERROR: No RGB/Depth pairs found ===================")
        print("Check the file naming convention matches that specified in the dataGen.py functions")
        return
    
    #Split into training and validation. Test should be split manually.
    numTrain = int(math.floor(trainFrac*len(rgbFiles)))
    numVal = len(rgbFiles)-numTrain
    
    #Make the dictionaries
    labels = {}
    partition = {}
    
    index = 0
    for index in range(0,len(rgbFiles)):
        labels[rgbFiles[index]] = depthFiles[index]
        
    partition['train'] = rgbFiles[:numTrain]
    partition['validation'] = rgbFiles[-numVal:]
    
    print("A total of {} RGB/Depth pairs will be used for training and {} for validation".format(numTrain, numVal))
    
    #Partition now looks like
    # {'train': ['file1', 'file2', 'file3'], 'validation': ['file4']}
    
    return partition, labels

def buildKittiList(rgbDataDir, depthDataDir, trainFrac=0.8):
    '''Builds a list of all the RGB/Depth image pairs from the kitti dataset,
    use the dataset bash scripts to generate the correct images from raw kitti data'''
    path = os.path.join(rgbDataDir, '*.png')
    rgbFiles = glob.glob(path)
    depthFiles = []
    
    print("A total of {} RGB files were found".format(len(rgbFiles)))

    #Match each RGB image to a depth image
    for rgbFileCurrent in rgbFiles:
        rgbFile = os.path.basename(rgbFileCurrent)
        
        #This should match the file naming convention
        rgbFile = rgbFile.replace("image", "groundtruth_depth", 1)
        
        path = os.path.join(depthDataDir, rgbFile)
        if(os.path.isfile(path)):
            depthFiles.append(path)
        else:
            rgbFiles.remove(rgbFileCurrent)
            print("Warning: A matching depth image could not be found for the file:")
            print(rgbFile)
    
    if(len(depthFiles) < 1):
        print("=================== ERROR: No RGB/Depth pairs found ===================")
        print("Check the file naming convention matches that specified in the dataGen.py functions")
        return
    
    #Split into training and validation. Test should be split manually.
    numTrain = int(math.floor(trainFrac*len(rgbFiles)))
    numVal = len(rgbFiles)-numTrain
    
    #Make the dictionaries
    labels = {}
    partition = {}
    
    index = 0
    for index in range(0,len(rgbFiles)):
        labels[rgbFiles[index]] = depthFiles[index]
        
    partition['train'] = rgbFiles[:numTrain]
    partition['validation'] = rgbFiles[-numVal:]
    
    print("A total of {} RGB/Depth pairs will be used for training and {} for validation".format(numTrain, numVal))
    
    #Partition now looks like
    # {'train': ['file1', 'file2', 'file3'], 'validation': ['file4']}
    
    return partition, labels
    
def buildWashingtonList(rgbDataDir, depthDataDir, trainFrac=0.8):
    '''Builds a list of all the RGB/Depth image pairs from the washington RGBD dataset,
    use the dataset bash scripts to split the images into different directories'''
    path = os.path.join(rgbDataDir, '*.png')
    rgbFiles = glob.glob(path)
    depthFiles = []
    
    print("A total of {} RGB files were found".format(len(rgbFiles)))

    #Match each RGB image to a depth image
    for rgbFileCurrent in rgbFiles:
        rgbFile = os.path.basename(rgbFileCurrent)
        
        #This should match the file naming convention
        filename, ext = os.path.splitext(rgbFile)
        rgbFile = filename + '_depth' + ext
        
        path = os.path.join(depthDataDir, rgbFile)
        if(os.path.isfile(path)):
            depthFiles.append(path)
        else:
            rgbFiles.remove(rgbFileCurrent)
            print("Warning: A matching depth image could not be found for the file:")
            print(rgbFile)
    
    if(len(depthFiles) < 1):
        print("=================== ERROR: No RGB/Depth pairs found ===================")
        print("Check the file naming convention matches that specified in the dataGen.py functions")
        return
    
    #Split into training and validation. Test should be split manually.
    numTrain = int(math.floor(trainFrac*len(rgbFiles)))
    numVal = len(rgbFiles)-numTrain
    
    #Make the dictionaries
    labels = {}
    partition = {}
    
    index = 0
    for index in range(0,len(rgbFiles)):
        labels[rgbFiles[index]] = depthFiles[index]
        
    partition['train'] = rgbFiles[:numTrain]
    partition['validation'] = rgbFiles[-numVal:]
    
    print("A total of {} RGB/Depth pairs will be used for training and {} for validation".format(numTrain, numVal))
    
    #Partition now looks like
    # {'train': ['file1', 'file2', 'file3'], 'validation': ['file4']}
    
    return partition, labels

    
if __name__ == "__main__":
#    rgbDataDir = "/home/jasper/Downloads/TestKittiDataset/RGBs"
#    depthDataDir = "/home/jasper/Downloads/TestKittiDataset/depths"
#    part, labels = buildKittiList(rgbDataDir, depthDataDir)
#    
#    ycbDir = "/media/jasper/107480BB7480A4D6/Users/Jasper/Datasets/YCB"
#    part, labels = buildYCBList(ycbDir)
    
    redwoodDir = "/media/jasper/107480BB7480A4D6/Users/Jasper/Datasets/Redwood"
    part, labels = buildRedwoodList(redwoodDir)
    
#    kittiGen = KittiDataGenerator(part['train'], labels)
    
#    rgbDataDir = "/home/jasper/Downloads/rgbd-dataset/apple/apple_1/images"
#    depthDataDir = "/home/jasper/Downloads/rgbd-dataset/apple/apple_1/depth"
#    part, labels = buildWashingtonList(rgbDataDir, depthDataDir, 0.2)
#    
#    washingtonGen = RGBDDataGenerator(part['train'], labels)