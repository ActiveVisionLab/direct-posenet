import os
import os.path as osp
import numpy as np
import torch
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

sys.path.insert(0, '../../')
from seven_scenes import SevenScenes
from mpl_toolkits.mplot3d import Axes3D

# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d # rays_o (100,100,3), rays_d (100,100,3)

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2] # t_n
    rays_o = rays_o + t[...,None] * rays_d # here rays_o[...,2] must be -1 (-near)

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1) # o'
    rays_d = torch.stack([d0,d1,d2], -1) # d'

    return rays_o, rays_d

def get_pose_from_dataset(seq, mode, num_workers,trainskip, bs, isTrainset, data_dir='../data/deepslam_data/7Scenes'):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if isTrainset:
        dset = SevenScenes(seq, data_dir, isTrainset, transform, mode=mode, trainskip=trainskip, half_res=True)
    else:
        dset = SevenScenes(seq, data_dir, isTrainset, transform, mode=mode, testskip=trainskip, half_res=True)

    print('Loaded 7Scenes sequence {:s}, length = {:d}'.format(seq, len(dset)))
    data_loader = DataLoader(dset, batch_size=bs, shuffle=False,)
    tpose = np.zeros((len(dset),3,4))
    for batch_idx, (data, target) in enumerate(data_loader):
        istart=batch_idx*bs
        iend = (batch_idx+1)*bs
        if batch_idx % 200 == 0:
            print ('Image {:d} / {:d}'.format(batch_idx, len(dset)))
        #rgb = data[:bs]
        pose = target[:bs]
        tpose[istart:iend] = pose.reshape(bs,3,4)

    x = tpose[:,0,3]
    y = tpose[:,1,3]
    z = tpose[:,2,3]
    return x, y, z, tpose, dset, data_loader

def rescale(x, y, z, tpose, sc):
    ''' rescale camera translational pose '''
    x_norm = x*sc
    y_norm = y*sc
    z_norm = z*sc

    # update pose

    tpose[:,0,3] = x_norm
    tpose[:,1,3] = y_norm
    tpose[:,2,3] = z_norm
    return tpose, x_norm, y_norm, z_norm

def llff_recenter(x, y, z, tpose, sc):
    def normalize(x):
        return x / np.linalg.norm(x)
    def viewmatrix(z, up, pos):
        vec2 = normalize(z)
        vec1_avg = up
        vec0 = normalize(np.cross(vec1_avg, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    ### normalize xyz into [-1, 1] ###

    # determine scale, 
    x_norm = x*sc
    y_norm = y*sc
    z_norm = z*sc

    tpose_ = tpose+0
    tpose_[:,0,3] = x_norm
    tpose_[:,1,3] = y_norm
    tpose_[:,2,3] = z_norm

    # recenter pose
    center = np.array([x_norm.mean(), y_norm.mean(), z_norm.mean()])
    bottom = np.reshape([0,0,0,1.], [1,4])

    # pose avg
    vec2 = normalize(tpose_[:, :3, 2].sum(0)) # array([-0.30415403,  0.43312491,  0.84846517])
    up = tpose_[:, :3, 1].sum(0) # array([ 135.47424833,  879.29250356, -267.50269427])
    hwf=np.array([[240.,320.,279.0]]).transpose()

    # 279.0 240 320

    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    print('c2w:', c2w)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [tpose_.shape[0],1,1])
    poses = np.concatenate([tpose_[:,:3,:4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses

    x_recentered=poses[:,0,3]
    y_recentered=poses[:,1,3]
    z_recentered=poses[:,2,3]

    return poses, x_recentered, y_recentered, z_recentered



def draw_voxel(pts):
    # # initialize voxel tabel
    voxel_cube = np.zeros((20,20,20), dtype=bool)
    #print("number of rays for this view: ", pts.shape[0])
    for ray_i in range (pts.shape[0]):
        # transfer [-1,1] space to [0,20]
        pts_np = pts[ray_i,:,:].numpy()
        xi = np.array(pts_np[:,0])
        yi = np.array(pts_np[:,1])
        zi = np.array(pts_np[:,2])

        s = pts.shape[1] # samples of each ray

        # round to .1 decimal
        for i in range(s):
            xi[i] = round(xi[i],1)
            yi[i] = round(yi[i],1)
            zi[i] = round(zi[i],1)

        xi = (xi*10).astype(int)+10
        yi = (yi*10).astype(int)+10
        zi = (zi*10).astype(int)+10
        
        xi = np.clip(xi, 0,19) # set min val 0, max val 19
        yi = np.clip(yi, 0,19)
        zi = np.clip(zi, 0,19)

        # update voxel cube table
        voxel_cube[xi,yi,zi] = True
    
    count = np.count_nonzero(voxel_cube)
    print("Total voxel volume: {}".format(count))
#     # combine the objects into a single boolean array
#     voxels = voxel_cube #| cube2 #| link

#     # set the colors of each object
#     colors = np.empty(voxels.shape, dtype=object)
#     #colors[link] = 'red'
#     colors[voxel_cube] = 'blue'
#     #colors[cube2] = 'green'

#     # and plot everything
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.voxels(voxels, facecolors=colors, edgecolor='k')

#     plt.show()
    return voxel_cube, count

def pose_to_rays(pose, H, W, focal, near, far, ndc=False, lindisp=False):
    ''' mimic nerf ray sampling, convert pose to rays '''
    rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))

    # pick 1024 rays
    N_rand = 5
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
    select_coords = coords[select_inds].long()  # (N_rand, 2)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)
    
    # use provided ray batch in render()
    rays_o, rays_d = batch_rays
    if 1: # If True, use viewing direction of a point in space in model.
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
    sh = rays_d.shape

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if 1:
        rays = torch.cat([rays, viewdirs], -1) # [1024, 11]
    
    # in Batchify(): render_rays()
    ray_batch = rays
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    
    N_samples = 64
    t_vals = torch.linspace(0., 1., steps=N_samples)
    
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals) # sample in depth
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals)) # sample in diparity disparity = 1/depth

    z_vals = z_vals.expand([N_rays, N_samples])
    
    if 1: #perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # find mid points of each interval
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape) # randomly choose in intervals

        z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # Sample 3D point print it out, [batchsize, 64 samples, 3 axis]
    return pts

def save_frustum(img_file, voxel_cube):
    ''' Save frustum into files '''
    # Parse color img filenames
    data_dir, filename = osp.split(img_file) #'../data/deepslam_data/7Scenes/heads/seq-02', 'frame-000000.color.png'
    scene_dir, seq_name = osp.split(data_dir) #'../data/deepslam_data/7Scenes/heads', 'seq-02'
    dataset_dir, scene = osp.split(scene_dir) # '../data/deepslam_data/7Scenes', 'heads'

    # create a new folder to store frustum file
    frustum_dir = osp.join('../data/7Scenes_frustum', scene, seq_name)
    os.makedirs(frustum_dir, exist_ok=True)

    # Frustum save filename
    frustum_filename = osp.join(frustum_dir, filename[:-10]+'.frustum.npy')
    np.save(frustum_filename, voxel_cube)