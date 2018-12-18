# Author: Jasper Brown
# jasperebrown@gmail.com
# 2019

import numpy as np
import cv2
from open3d import *
import matplotlib.pyplot as plt

# Uses Open3D: www.open3d.org
color_raw = read_image("/home/jasper/git/ImageDepthPredictionKeras/SampleData/rgb4.jpg")
depth_raw = read_image("/home/jasper/git/ImageDepthPredictionKeras/SampleData/depth4.png")
rgbd_image = create_rgbd_image_from_color_and_depth(
    color_raw, depth_raw);
        
print(rgbd_image)

#display RGBD
plt.subplot(1, 2, 1)
plt.title('Redwood grayscale image')
plt.imshow(rgbd_image.color)
plt.subplot(1, 2, 2)
plt.title('Redwood depth image')
plt.imshow(rgbd_image.depth)
plt.show()

#Display point cloud
pcd = create_point_cloud_from_rgbd_image(rgbd_image, PinholeCameraIntrinsic(
        PinholeCameraIntrinsicParameters.PrimeSenseDefault))

# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
draw_geometries([pcd])