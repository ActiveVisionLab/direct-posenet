import set_sys_path
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm, trange
from torchsummary import summary
import matplotlib.pyplot as plt

from pose_model import *
from direct_pose_model import *
from callbacks import EarlyStopping
from run_nerf_helpers import *
from prepare_data import prepare_data, load_dataset
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader

import pdb


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    parser.add_argument("--multi_gpu", action='store_true', help='use multiple gpu on the server')

    # 7Scenes
    parser.add_argument("--trainskip", type=int, default=1, help='will load 1/N images from train sets, useful for large datasets like 7 Scenes')
    parser.add_argument("--use_poses_bounds", action='store_true', help='use poses bounds that came from COLMAP like LLFF dataset')
    parser.add_argument("--use_ndc_7Scenes", action='store_true', help='use ndc flag for 7Scenes dataset')
    parser.add_argument("--pose_scale", type=int, default=1, help='manual tuned pose scale factor, matching COLMAP result')
    parser.add_argument("--near_far", nargs='+', type=float, default=[0.5, 2.5], help='setting near, far params, NDC [0., 1.], no NDC [0.5, 2.5]')
    parser.add_argument("--use_bounds", action='store_true', help='use pre-computed bounds from depth maps') # currently didn't use this
    parser.add_argument("--df", type=int, default=2, help='image downscale factor')
    parser.add_argument("--reduce_embedding", type=int, default=-1, help='fourier embedding mode: -1: paper default, \
                                                                        0: reduce by half, 1: remove embedding, 2: DNeRF embedding')
    parser.add_argument("--epochToMaxFreq", type=int, default=-1, help='DNeRF embedding mode: (based on DNeRF paper): \
                                                                        hyper-parameter for when α should reach the maximum number of frequencies m')
    parser.add_argument("--render_pose_only", action='store_true', help='render a spiral video for 7 Scene')
    parser.add_argument("--save_pose_avg_stats", action='store_true', help='save a pose avg stats to unify NeRF, posenet, nerf tracking training')
    parser.add_argument("--load_pose_avg_stats", action='store_true', help='load precomputed pose avg stats to unify NeRF, posenet, nerf tracking training')
    parser.add_argument("--finetune_unlabel", action='store_true', help='finetune unlabeled sequence like MapNet')
    parser.add_argument("--i_eval",   type=int, default=50, help='frequency of eval posenet result')
    parser.add_argument("--save_all_ckpt", action='store_true', help='save all ckpts for each epoch')
    parser.add_argument("--train_local_nerf", type=int, default=-1, help='train local NeRF with ith training sequence only, ie. Stairs can pick 0~3')
    parser.add_argument("--render_video_train", action='store_true', help='render train set NeRF and save as video, make sure i_eval is True')
    parser.add_argument("--render_video_test", action='store_true', help='render val set NeRF and save as video,  make sure i_eval is True')
    parser.add_argument("--no_DNeRF_viewdir", action='store_true', default=False, help='will not use DNeRF in viewdir encoding')

    ##################### PoseNet Settings ########################
    parser.add_argument("--pose_only", type=int, default=0, help='posenet type to train, \
                        1: train baseline posenet, 2: posenet+nerf manual optimize, 3: posenet+nerf easy optimize')
    parser.add_argument("--learning_rate", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--batch_size", type=int, default=1, help='train posenet only')
    parser.add_argument("--pretrain_model_path", type=str, default='', help='model path of pretrained model')
    parser.add_argument("--model_name", type=str, help='pose model output folder name')
    parser.add_argument("--combine_loss", action='store_true',
                        help='combined l2 pose loss + rgb mse loss')
    parser.add_argument("--combine_loss_w", nargs='+', type=float, default=[0.5, 0.5], 
                        help='weights of combined loss ex, [0.5 0.5], \
                        default None, only use when combine_loss is True')
    parser.add_argument("--patience", nargs='+', type=int, default=[200, 50], help='set training schedule for patience [EarlyStopping, reduceLR]')
    parser.add_argument("--resize_factor", type=int, default=2, help='image resize downsample ratio')
    parser.add_argument("--freezeBN", action='store_true', help='Freeze the Batch Norm layer at training PoseNet')
    parser.add_argument("--preprocess_ImgNet", action='store_true', help='Normalize input data for PoseNet')
    parser.add_argument("--eval", action='store_true', help='eval model')
    parser.add_argument("--no_save_multiple", action='store_true', help='default, save multiple posenet model, if true, save only one posenet model')
    parser.add_argument("--resnet34", action='store_true', default=False, help='use resnet34 backbone instead of mobilenetV2')
    parser.add_argument("--dropout", type=float, default=0.5, help='dropout rate for resnet34 backbone')


    ##################### NeRF Settings ########################
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # legacy mesh options
    parser.add_argument("--mesh_only", action='store_true', help='do not optimize, reload weights and save mesh to a file')
    parser.add_argument("--mesh_grid_size", type=int, default=80,help='number of grid points to sample in each dimension for marching cubes')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1, help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## legacy blender flags
    parser.add_argument("--white_bkgd", action='store_true', help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,  help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, help='will take every 1/N images as LLFF test set, paper uses 8')
    parser.add_argument("--no_bd_factor", action='store_true', default=False, help='do not use bd factor')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, help='frequency of render_poses video saving')

    # quaternion experiment, remove later
    parser.add_argument("--quat_exp", action='store_true', default=False, help='input quaternion for direct-posenet')

    return parser

parser = config_parser()
args = parser.parse_args()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # You can specify a GPU by CUDA_VISIBLE_DEVICES=2 python run_nerf.py ...
device = torch.device('cuda:0') # this is really controlled in train.sh

def freeze_bn_layer(model):
    print("Freezing BatchNorm Layers...")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # print("this is a BN layer:", module)
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
    return model

def render_test(args, train_dl, val_dl, hwf, start, model, device, render_kwargs_test):
    model.eval()

    # ### Eval Training set result
    if args.render_video_train:
        images_train = []
        poses_train = []
        # views from train set
        for img, pose in train_dl:
            predict_pose = inference_pose_regression(args, img, device, model)
            device_cpu = torch.device('cpu')
            predict_pose = predict_pose.to(device_cpu) # put predict pose back to cpu

            img_val = img.permute(0,2,3,1) # (1,240,320,3)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = predict_pose.reshape(3,4)[:3,:4] # (1,3,4))
            pose_val[0,3,3] = 1.
            images_train.append(img_val)
            poses_train.append(pose_val)

        images_train = torch.cat(images_train, dim=0).numpy()
        poses_train = torch.cat(poses_train, dim=0)
        print('train poses shape', poses_train.shape)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        with torch.no_grad():
            rgbs, disps = render_path(poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=None)
        torch.set_default_tensor_type('torch.FloatTensor')
        print('Saving trainset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.model_name, '{}_trainset_{:06d}_'.format(args.model_name, start))
        imageio.mimwrite(moviebase + 'train_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'train_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)

    ### Eval Validation set result
    if args.render_video_test:
        images_val = []
        poses_val = []
        # views from val set
        for img, pose in val_dl: 
            predict_pose = inference_pose_regression(args, img, device, model)
            device_cpu = torch.device('cpu')
            predict_pose = predict_pose.to(device_cpu) # put predict pose back to cpu

            img_val = img.permute(0,2,3,1) # (1,240,360,3)
            pose_val = torch.zeros(1,4,4)
            pose_val[0,:3,:4] = predict_pose.reshape(3,4)[:3,:4] # (1,3,4))
            pose_val[0,3,3] = 1.
            images_val.append(img_val)
            poses_val.append(pose_val)

        images_val = torch.cat(images_val, dim=0).numpy()
        poses_val = torch.cat(poses_val, dim=0)
        print('test poses shape', poses_val.shape)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        with torch.no_grad():
            rgbs, disps = render_path(poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=None)
        torch.set_default_tensor_type('torch.FloatTensor')
        print('Saving testset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.model_name, '{}_test_{:06d}_'.format(args.model_name, start))
        imageio.mimwrite(moviebase + 'test_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'test_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    return

def train():
    print(parser.format_values())

    # Load data
    if args.dataset_type == '7Scenes':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
    else:
        images, poses_train, render_poses, hwf, i_split, near, far = load_dataset(args)
        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

    POSENET=args.pose_only
    if POSENET==1: # Step1: Pretrain PoseNet Only, Input a image, Output rotation matrix
        # Prepare dataloaders for PoseNet, each batch contains (image, pose)
        if args.dataset_type == 'blender' or args.dataset_type == 'llff':
            train_dl, val_dl, test_dl = prepare_data(args, images, poses_train, i_split)
        
        # create model
        if args.resnet34: # for paper experiment table1
            model = PoseNet_res34(droprate=args.dropout)
        else: # default mobilenetv2 backbone
            model = PoseNetV2()
        # Freeze BN to not updating gamma and beta
        if args.freezeBN:
            model = freeze_bn_layer(model)
        model.to(device)

        # print model structure
        #summary(model, (3, 200, 200)) # Must set to device cuda:0 to use it

        # set loss
        loss = nn.MSELoss()

        # set optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #weight_decay=weight_decay, **kwargs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=args.patience[1], verbose=True)

        # set callbacks parameters
        early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False)

        # start training
        train_posenet(args, train_dl, val_dl, model, 2000, optimizer, loss, scheduler, device, early_stopping)

    elif POSENET==2: # Step2: Finetune for Direct-PN
        if args.pretrain_model_path == '':
            print('training PoseNet from scratch')
            model = PoseNetV2()
            model = freeze_bn_layer(model)
        else:
            # load pretrained PoseNet model
            model = load_exisiting_model(args)
        if args.freezeBN:
            model = freeze_bn_layer(model)
        model.to(device)
        #summary(model, (3, 240, 320)) # Must set to device cuda:0 to use it

        # set optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #weight_decay=weight_decay, **kwargs
        
        # set callbacks parameters
        early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False)
        # start training
        if args.dataset_type == '7Scenes':
            train_nerf_tracking(args, model, optimizer, i_split, hwf, near, far, device, early_stopping, train_dl=train_dl, val_dl=val_dl, test_dl=test_dl)
        else: # blender/llff dataset
            train_nerf_tracking(args, model, optimizer, i_split, hwf, near, far, device, early_stopping, images=images, poses_train=poses_train)

def eval():
    print(parser.format_values())
    # Load data
    if args.dataset_type == '7Scenes':
        train_dl, val_dl, test_dl, hwf, i_split, near, far = load_7Scenes_dataloader(args)
    else:
        images, poses_train, render_poses, hwf, i_split, near, far = load_dataset(args)
        i_train, i_val, i_test = i_split
        # Prepare dataloaders for PoseNet, each batch contains (image, pose)
        train_dl, val_dl, test_dl = prepare_data(args, images, poses_train, i_split)
    
    # load pretrained PoseNet model
    model = load_exisiting_model(args)
    if args.freezeBN:
            model = freeze_bn_layer(model)
    model.to(device)

    print(len(val_dl.dataset))

    if args.render_video_train or args.render_video_test: # save render video 
        render_kwargs_train, render_kwargs_test, start, _, _ = create_nerf_7Scenes(args)
        bds_dict = {
            'near' : near,
            'far' : far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)
        if args.reduce_embedding==2:
            render_kwargs_test['i_epoch'] = start
        with torch.no_grad():
            render_test(args, train_dl, val_dl, hwf, start, model, device, render_kwargs_test)
        return

    # Todo: remove TF code here
    # get_error_in_q(args, train_dl, model, len(train_dl.dataset), device, batch_size=args.batch_size)
    get_error_in_q(args, val_dl, model, len(val_dl.dataset), device, batch_size=1)
    
    if 0: # print some render result
        render_kwargs_train, render_kwargs_test, start, _, _ = create_nerf_7Scenes(args)
        bds_dict = {
            'near' : near,
            'far' : far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)
        if args.reduce_embedding==2:
            render_kwargs_test['i_epoch'] = start
        save_val_result_7Scenes(args, 0, val_dl, model, hwf, True, device, num_samples=10, **render_kwargs_test)


if __name__ == '__main__':
    if args.eval:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        eval()
    else:
        train()

