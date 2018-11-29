import os
from shutil import copyfile
import numpy as np
import glob


dataDir = '/media/jasper/bigData1/redwoodDataset'
destDir = '/mnt/0FEF1F423FF4C54B/Datasets/Redwood'
ratio = 25

os.chdir(dataDir)

for root, dir,files in os.walk("."):  
    #if in the RGB folder, select every 1 in 'ratio' image and 
    # find a matching depth image based on timestamp
    if(os.path.split(root)[1] == 'rgb'):
        count = 0
        rgbFiles = glob.glob(root + '/*.jpg')
        rgbFiles = sorted(rgbFiles)
        
        print("Processing dir " + root)
        
        depthFilesAvail = glob.glob(root + '/../depth/*.png')
        depthFilesAvail = sorted(depthFilesAvail)
        depthTimes = np.zeros(len(depthFilesAvail))
        
        for depthIndex in range(0,len(depthFilesAvail)):
            _,depthTime = os.path.split(depthFilesAvail[depthIndex])
            depthTime,_ = os.path.splitext(depthTime)
            #Split at hyphen to get microseconds
            depthTime = int(depthTime.split('-')[1])
            depthTimes[depthIndex] = depthTime
        
        for file in rgbFiles:
            if(count%ratio == 0):
                srcPath = file
                
                rgbTime,_ = os.path.splitext(file)
                rgbTime = int(rgbTime.split('-')[1])
                depthID = (np.abs(depthTimes - rgbTime)).argmin()
                if(abs(rgbTime - depthTimes[depthID]) > 100000):
                    print("Skipping image, no matching depth frame within 0.1 seconds of timestamp:")
                    print(srcPath)
                    break
                    
                destPath = srcPath[1:] #Drop the dot
                destPath = destDir + destPath
                
                if(not os.path.isdir(os.path.split(destPath)[0])):
                    os.makedirs(os.path.split(destPath)[0])
                
                #Copy the rgb file
                copyfile(srcPath, destPath)
                
                srcPath = depthFilesAvail[depthID]
                destPath = srcPath[1:] #Drop the dot
                destPath = destDir + destPath
                
                if(not os.path.isdir(os.path.split(destPath)[0])):
                    os.makedirs(os.path.split(destPath)[0])
                    
                #Copy the depth file
                copyfile(srcPath, destPath)
                
            count = count + 1