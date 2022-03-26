import os
import os.path as osp
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch.cuda
from torch.utils.data import DataLoader
from torchvision import transforms, models

from dataset_loaders.seven_scenes import SevenScenes, normalize_recenter_pose, load_image

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

# translation z axis
trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).astype(float)

# x rotation
rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).astype(float)

# y rotation
rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).astype(float)

# z rotation
rot_psi = lambda psi : np.array([
    [np.cos(psi),-np.sin(psi),0,0],
    [np.sin(psi),np.cos(psi),0,0],
    [0,0,1,0],
    [0,0,0,1]]).astype(float)

def camera_frustum_initializer(args):
    ''' initialize camera frustum parameters '''
    NEAR_THRESH = args.near_far[0]
    FAR_THRESH = args.near_far[1]
    SAMPLE_STEP = 0.1

    SCENES7_X_RES = 640.0/2
    SCENES7_Y_RES = 480.0/2
    SCENES7_F = 585.0/2 # 585/2
    SCENES7_CX = SCENES7_X_RES / 2.0
    SCENES7_CY = SCENES7_Y_RES / 2.0

    W = 640.0/2
    H = 480.0/2
    K_APPROX = initK(SCENES7_F, SCENES7_CX, SCENES7_CY)

    FRUSTUM_APPROX = generate_sampling_frustum(
        SAMPLE_STEP, FAR_THRESH, K_APPROX, SCENES7_F, SCENES7_CX, SCENES7_CY, SCENES7_X_RES, SCENES7_Y_RES
    )
    return  K_APPROX, FRUSTUM_APPROX, W, H

def is_inside_frustum(p, x_res, y_res):
    return (0 < p[0]) & (p[0] < x_res) & (0 < p[1]) & (p[1] < y_res)

def initK(f, cx, cy):
    K = np.eye(3, 3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = cx
    K[1, 2] = cy
    return K

def generate_sampling_frustum(step, depth, K, f, cx, cy, x_res, y_res):
    x_max = depth * (x_res - cx) / f
    x_min = -depth * cx / f
    y_max = depth * (y_res - cy) / f
    y_min = -depth * cy / f

    zs = np.arange(0, depth, step)
    xs = np.arange(x_min, x_max, step)
    ys = np.arange(y_min, y_max, step)

    X0 = []

    for z in zs:
        for x in xs:
            for y in ys:
                P = np.array([x, y, z])
                p = np.dot(K, P)
                if p[2] < 0.00001:
                    continue
                p = p / p[2]
                if is_inside_frustum(p, x_res, y_res):
                    X0.append(P)
    X0 = np.array(X0)
    return X0

def compute_frustums_overlap(pose0, pose1, sampling_frustum, K, x_res, y_res):
    R0 = pose0[0:3, 0:3]
    t0 = pose0[0:3, 3]
    R1 = pose1[0:3, 0:3]
    t1 = pose1[0:3, 3]

    R10 = np.dot(R1.T, R0)
    t10 = np.dot(R1.T, t0 - t1)

    _P = np.dot(R10, sampling_frustum.T).T + t10
    p = np.dot(K, _P.T).T
    pn = p[:, 2]
    p = np.divide(p, pn[:, None])
    res = np.apply_along_axis(is_inside_frustum, 1, p, x_res, y_res)
    return np.sum(res) / float(res.shape[0])

def perturb_rotation(c2w, theta, phi, psi=0):
    last_row = np.tile(np.array([0, 0, 0, 1]), (1, 1))  # (N_images, 1, 4)
    c2w = np.concatenate([c2w, last_row], 0)  # (N_images, 4, 4) homogeneous coordinate
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_psi(psi/180.*np.pi) @ c2w
    c2w = c2w[:3,:4]
    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)
    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)
    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)
    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)
    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg

def center_poses(poses, pose_avg_from_file=None):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)
        pose_avg_from_file: if not None, pose_avg is loaded from pose_avg_stats.txt

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """


    if pose_avg_from_file is None:
        pose_avg = average_poses(poses)  # (3, 4) # this need to be fixed throughout dataset
    else:
        pose_avg = pose_avg_from_file

    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation (4,4)
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg #np.linalg.inv(pose_avg_homo)

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5] # it's empty here...
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def average_poses(poses):
    """
    Same as in SingleCamVideoStatic.py
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    center = poses[..., 3].mean(0)  # (3)
    z = normalize(poses[..., 2].mean(0))  # (3)
    y_ = poses[..., 1].mean(0)  # (3)
    x = normalize(np.cross(y_, z))  # (3)
    y = np.cross(z, x)  # (3)
    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
    return pose_avg

def generate_render_pose(poses, bds):
    idx = np.random.choice(poses.shape[0])
    c2w=poses[idx]
    print(c2w[:3,:4])
    
    ## Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min()*.9, bds.max()*5.
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 20, 0) # views of 20 degrees
    c2w_path = c2w
    N_views = 120 # number of views in video
    N_rots = 2

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    return render_poses

def perturb_render_pose(poses, bds, x, angle):
    """
    Inputs:
        poses: (3, 4)
        bds: bounds
        x: translational perturb range
        angle: rotation angle perturb range in degrees
    Outputs:
        new_c2w: (N_views, 3, 4) new poses
    """
    idx = np.random.choice(poses.shape[0])
    c2w=poses[idx]
    
    N_views = 10 # number of views in video    
    new_c2w = np.zeros((N_views, 3, 4))

    # perturb translational pose
    for i in range(N_views):
        new_c2w[i] = c2w
        new_c2w[i,:,3] = new_c2w[i,:,3] + np.random.uniform(-x,x,3) # perturb pos between -1 to 1
        theta=np.random.uniform(-angle,angle,1) # in degrees
        phi=np.random.uniform(-angle,angle,1) # in degrees
        psi=np.random.uniform(-angle,angle,1) # in degrees
        new_c2w[i] = perturb_rotation(new_c2w[i], theta, phi, psi)
    return new_c2w, idx

def remove_overlap_data(train_set, val_set):
    ''' Remove some overlap data in val set so that train set and val set do not have overlap '''
    train = train_set.gt_idx
    val = val_set.gt_idx

    # find redundant data index in val_set
    index = np.where(np.in1d(val, train) == True) # this is a tuple
    # delete redundant data
    val_set.gt_idx = np.delete(val_set.gt_idx, index)
    val_set.poses = np.delete(val_set.poses, index, axis=0)
    for i in sorted(index[0], reverse=True):
        val_set.c_imgs.pop(i) 
        val_set.d_imgs.pop(i)
    return train_set, val_set

def fix_coord(args, train_set, val_set, pose_avg_stats_file=''):
    ''' fix coord for 7 Scenes to align with llff style dataset '''
    # get all poses (train+val)
    train_poses = train_set.poses
    val_poses = val_set.poses
    all_poses = np.concatenate([train_poses, val_poses])

    # Center the poses for ndc
    all_poses = all_poses.reshape(all_poses.shape[0], 3, 4)

    # Here we use either pre-stored pose average stats or calculate pose average stats on the flight to center the poses
    if args.load_pose_avg_stats:
        pose_avg_from_file = np.loadtxt(pose_avg_stats_file)
        all_poses, pose_avg = center_poses(all_poses, pose_avg_from_file)
    else:
        all_poses, pose_avg = center_poses(all_poses)
    
    # This is only to store a pre-calculated pose average stats of the dataset
    if args.save_pose_avg_stats:
        if pose_avg_stats_file == '':
            print('pose_avg_stats_file location unspecified, please double check...')
            sys.exit()
        # save pose_avg to pose_avg_stats.txt
        np.savetxt(pose_avg_stats_file, pose_avg)
        print('pose_avg_stats.txt successfully saved')
        sys.exit()
    
    ### args.fix_coord, obsolete flag
    # Correct axis to LLFF Style y,z -> -y,-z
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(all_poses), 1, 1))  # (N_images, 1, 4)
    all_poses = np.concatenate([all_poses, last_row], 1)

    # correct rotation matrix from "up left forward" to "up right backward"
    flip_M = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]) # Mirror matrix that flip y & z direction
    flip_M = np.repeat(flip_M[None,:], all_poses.shape[0], axis=0)

    # all_poses = flip_M@all_poses@flip_M.transpose((0,2,1)) # This is correct M*[R|T]*M.T
    all_poses = flip_M@(all_poses@flip_M) # bug here M*([R|T]*M)
    all_poses = all_poses[:,:3,:4]

    bounds = np.array([args.near_far[0], args.near_far[1]]) # manual tuned, currently not used. Tune near-far instead

    sc=args.pose_scale # manual tuned factor
    all_poses[:,:3,3] *= sc

    # Return all poses to dataset loaders
    all_poses = all_poses.reshape(all_poses.shape[0], 12)
    train_set.poses = all_poses[:train_poses.shape[0]]
    val_set.poses = all_poses[train_poses.shape[0]:]
    return train_set, val_set, bounds

def fetch_unique_view_index(args, train_set, threshold):
    ''' local NeRF data selection '''
    # Frustum near-far threshold
    K_APPROX, FRUSTUM_APPROX, W, H = camera_frustum_initializer(args)

    # compute how many unique views in the train set
    keyframe_idx = 0
    unique_frame_index = []
    unique_frame_index.append(keyframe_idx)

    for i in range(len(train_set)):
        if i % 200 == 0:
            print ('Image {:d} / {:d}'.format(i, len(train_set)))
        
        if keyframe_idx == i:
            continue
        
        # compute frustum overlap
        overlap_1p = compute_frustums_overlap(
            train_set.poses[keyframe_idx].reshape(3,4), train_set.poses[i].reshape(3,4), FRUSTUM_APPROX, K_APPROX, W, H
        )
        if overlap_1p > threshold:
            continue
            
        # double check with existing unique views
        save_flag = True
        for j in unique_frame_index:
            overlap_2p = compute_frustums_overlap(train_set.poses[j].reshape(3,4), train_set.poses[i].reshape(3,4), FRUSTUM_APPROX, K_APPROX, W, H)
            if overlap_2p > threshold:
                save_flag = False
                break

        if not save_flag:
            print('image[{}] is overlapped with unique_frame_index {}'.format(i, j))
            keyframe_idx = i
            continue

        unique_frame_index.append(i)
        print(unique_frame_index)
        keyframe_idx = i
    unique_frame_index = np.array(unique_frame_index)
    print('total train set selected', len(unique_frame_index))
    save_unique_view_dir = osp.join(args.datadir, 'unique_view.txt')
    np.savetxt(save_unique_view_dir, unique_frame_index, fmt="%d")
    return unique_frame_index

def select_nearest_neighbor_views(args, train_set, threshold, unique_frame_index):
    ''' select nearest_neighbor views based on unique_frame_index  '''
    # Frustum near-far threshold
    K_APPROX, FRUSTUM_APPROX, W, H = camera_frustum_initializer(args)

    # compute how many unique views in the train set
    keyframe_idx = 0
    frame_index = []

    for i in range(len(train_set)):
        if i % 200 == 0:
            print ('Image {:d} / {:d}'.format(i, len(train_set)))
        
        if keyframe_idx == i:
            continue
        
        # compute frustum overlap
        for j in unique_frame_index:
            overlap_2p = compute_frustums_overlap(train_set.poses[j].reshape(3,4), train_set.poses[i].reshape(3,4), FRUSTUM_APPROX, K_APPROX, W, H)
            if overlap_2p > threshold:
                frame_index.append(i)
                break
    frame_index = np.array(frame_index)
    frame_index = frame_index[::5]
    print('train set selected', len(frame_index))
    train_set.c_imgs = list(train_set.c_imgs[i] for i in frame_index)
    train_set.d_imgs = list(train_set.d_imgs[i] for i in frame_index)
    train_set.poses = train_set.poses[frame_index]
    return

def load_7Scenes_dataloader(args):
    ''' Data loader for Pose Regression Network '''
    if args.pose_only: # if train posenet is true
        pass
    else:
        raise Exception('load_7Scenes_dataloader() currently only support PoseNet Training, not NeRF training')
    ######## New Code under Construction ###########
    data_dir, scene = osp.split(args.datadir) # ../data/7Scenes, chess
    dataset_folder, dataset = osp.split(data_dir) # ../data, 7Scenes
    stats_filename = osp.join(args.datadir, 'stats.txt') # ../data/7Scenes/chess/stats.txt
    stats = np.loadtxt(stats_filename)

    # transformer
    data_transform = transforms.Compose([
        transforms.Resize(240),
        transforms.ToTensor(),
        ])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(args.datadir, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    data_dir = osp.join(dataset_folder, 'deepslam_data', dataset) # ../data/deepslam_data/7Scenes
    kwargs = dict(scene=scene, data_path=data_dir, transform=data_transform, target_transform=target_transform)

    if args.finetune_unlabel: # direct-pn + unlabel
        train_set = SevenScenes(train=False, half_res=args.half_res, testskip=args.trainskip, **kwargs)
        val_set = SevenScenes(train=False, half_res=args.half_res, testskip=args.testskip, **kwargs)
        
        if not args.eval:
            # remove overlap data in val_set that was already in train_set,
            train_set, val_set = remove_overlap_data(train_set, val_set)

    else:
        train_set = SevenScenes(train=True, half_res=args.half_res, trainskip=args.trainskip, **kwargs)
        val_set = SevenScenes(train=False, half_res=args.half_res, testskip=args.testskip, **kwargs)
    L = len(train_set)

    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # use a pose average stats computed earlier to unify posenet and nerf training
    if args.save_pose_avg_stats or args.load_pose_avg_stats:
        pose_avg_stats_file = osp.join(args.datadir, 'pose_avg_stats.txt')
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, pose_avg_stats_file)
    else:
        train_set, val_set, bounds = fix_coord(args, train_set, val_set)

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=5, pin_memory=True)
    val_dl = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = None # 7 Scenes don't have testset, validation set is the testset

    hwf = [train_set.H, train_set.W, train_set.focal]
    i_split = [i_train, i_val, i_test]

    return train_dl, val_dl, test_dl, hwf, i_split, bounds.min(), bounds.max()

def load_7Scenes_dataloader_NeRF(args):
    ''' Data loader for NeRF '''

    data_dir, scene = osp.split(args.datadir) # ../data/7Scenes, chess
    dataset_folder, dataset = osp.split(data_dir) # ../data, 7Scenes
    stats_filename = osp.join(args.datadir, 'stats.txt') # ../data/7Scenes/chess/stats.txt
    stats = np.loadtxt(stats_filename)

    data_transform = transforms.Compose([
        transforms.ToTensor()])
    target_transform = transforms.Lambda(lambda x: torch.Tensor(x))

    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(args.datadir, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    data_dir = osp.join('..', 'data', 'deepslam_data', dataset) # ../data/deepslam_data/7Scenes
    kwargs = dict(scene=scene, data_path=data_dir, mode=1, transform=data_transform, target_transform=target_transform, use_ndc=args.use_ndc_7Scenes, df=args.df)

    train_set = SevenScenes(train=True, half_res=args.half_res, trainskip=args.trainskip, **kwargs)
    val_set = SevenScenes(train=False, half_res=args.half_res, testskip=args.testskip, **kwargs)

    if args.train_local_nerf is not -1:
        N_partition = 4 # 1/N_partition unique views for local nerf
        # load or compute trainset unique view indexes of the scene
        if args.load_unique_view_stats:
            unique_frame_index = np.loadtxt(osp.join(args.datadir, 'unique_view.txt'), dtype=int)
        else:
            unique_frame_index = fetch_unique_view_index(args, train_set, args.frustum_overlap_th)
        num_unique_view = len(unique_frame_index)//N_partition # select Total unique views/N + 1 for each partition
        print('total unique_frame_index', unique_frame_index)
        
        # select the (train_local_nerf)th portion of training set
        start_i = args.train_local_nerf * num_unique_view
        end_i = start_i + num_unique_view + 1
        unique_frame_index = unique_frame_index[start_i:end_i]
        print('selected unique_frame_index', unique_frame_index)

        # select nearest neighbor frames
        select_nearest_neighbor_views(args, train_set, args.frustum_overlap_th, unique_frame_index)

    i_train = train_set.gt_idx
    i_val = val_set.gt_idx
    i_test = val_set.gt_idx

    # use a pose average stats computed earlier to unify posenet and nerf training
    if args.save_pose_avg_stats or args.load_pose_avg_stats:
        pose_avg_stats_file = osp.join(args.datadir, 'pose_avg_stats.txt')
        train_set, val_set, bounds = fix_coord(args, train_set, val_set, pose_avg_stats_file)
    else:
        train_set, val_set, bounds = fix_coord(args, train_set, val_set)

    render_poses = None
    render_img = None

    train_shuffle=True
    if args.render_video_train:
        train_shuffle=False
    train_dl = DataLoader(train_set, batch_size=1, shuffle=train_shuffle)
    val_dl = DataLoader(val_set, batch_size=1, shuffle=False)

    hwf = [train_set.H, train_set.W, train_set.focal]

    i_split = [i_train, i_val, i_test]
    
    return train_dl, val_dl, hwf, i_split, bounds, render_poses, render_img