### Import Libs ###
import os
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader

import sys
import pickle
import pdb
import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torchvision.utils import make_grid
import torchvision.transforms as transforms

from frustum_util import *

sys.path.insert(0, '../../')
from seven_scenes import SevenScenes

"""
visualizes the dataset
"""

seq = 'heads'
mode = 1
num_workers = 1
trainskip=1 # 4000/4
bs=1 # batch_size
isTrainset = True # if false, then it's valset

### Loop Thru Data ###
x, y, z, tpose, dset, data_loader = get_pose_from_dataset(seq, mode, num_workers,trainskip,bs, isTrainset)

### Rescale ###
sc = 0.5
tpose, x_norm, y_norm, z_norm = rescale(x, y, z, tpose, sc)

### Visualize Frustum ###
'''
1. Loop Thru all the given poses and converted to sample rays. Then save the frustum to files
2. Visualize frustum differences with the pivot image
'''
H = dset.H
W = dset.W
focal = dset.focal
near = 0
far = 1

# flags
compareflag = False
draw_compare = False
voxel_to_file = False
# end of flags

voxel_volume1 = 0
for i in range(tpose.shape[0]):
    # get ray sample points
    pts = pose_to_rays(tpose[i], H, W, focal, near, far)

    ### Save Frustum Voxels ###
    if voxel_to_file:
        voxel_cube, _ = draw_voxel(pts)
        save_frustum(dset.c_imgs[i], voxel_cube)

    ### Compare differences between 1st pose to the rest poses ###
    if draw_compare:
        if compareflag == False:
            first_vc, voxel_volume1 = draw_voxel(pts) # (20,20,20)
            compareflag = True
        else:
            second_vc, voxel_volume2 = draw_voxel(pts)
            voxel_volume_shared = np.count_nonzero(first_vc & second_vc)
            frustum_overlap_ratio = voxel_volume_shared / voxel_volume1
            print("frustum_overlap_ratio: {}".format(frustum_overlap_ratio)) 
            
            # combine the objects into a single boolean array
            voxels = first_vc | second_vc
            # set the colors of each object
            colors = np.empty(voxels.shape, dtype=object)
            colors[first_vc] = 'green'
            colors[second_vc] = 'blue'
            # and plot everything
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.voxels(voxels, facecolors=colors, edgecolor='k')
            plt.show()
print('done.')