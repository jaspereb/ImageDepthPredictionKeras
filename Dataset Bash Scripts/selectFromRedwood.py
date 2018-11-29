import os
from shutil import copyfile


dataDir = '/media/jasper/bigData1/redwoodDataset/first4'
destDir = '/mnt/0FEF1F423FF4C54B/Datasets/Redwood'
ratio = 50

os.chdir(dataDir)


for root, dir, files in os.walk("."):
    if(not files == []):
        print("Processing dir " + root)

    count = 0
    
    for file in files:
        if(count%ratio == 0):
            srcPath = os.path.join(root, file)
            count = count + 1
            
            destPath = srcPath[1:] #Drop the dot
            destPath = destDir + destPath
    
            copyfile(srcPath, destPath)