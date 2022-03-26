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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_nerf_helpers import *
from dataset_loaders.load_7Scenes import load_7Scenes_dataloader_NeRF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def run_network_DNeRF(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64, epoch=None, no_DNeRF_viewdir=False):
    """Prepares inputs and applies network 'fn'.
    """
    if epoch<0 or epoch==None:
        print("Error: run_network_DNeRF(): Invalid epoch")
        sys.exit()
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat, epoch)
    # add weighted function here
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        if no_DNeRF_viewdir:
            embedded_dirs = embeddirs_fn(input_dirs_flat)
        else:
            embedded_dirs = embeddirs_fn(input_dirs_flat, epoch)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def extract_mesh(render_kwargs, mesh_grid_size=80, threshold=50):
    network_query_fn, network = render_kwargs['network_query_fn'], render_kwargs['network_fine']
    device = next(network.parameters()).device

    with torch.no_grad():
        points = np.linspace(-1, 1, mesh_grid_size)
        query_pts = torch.tensor(np.stack(np.meshgrid(points, points, points), -1).astype(np.float32)).reshape(-1, 1, 3).to(device)
        viewdirs = torch.zeros(query_pts.shape[0], 3).to(device)

        output = network_query_fn(query_pts, viewdirs, network)

        grid = output[...,-1].reshape(mesh_grid_size, mesh_grid_size, mesh_grid_size)

        print('fraction occupied:', (grid > threshold).float().mean())

        vertices, triangles = mcubes.marching_cubes(grid.detach().cpu().numpy(), threshold)
        mesh = trimesh.Trimesh(vertices, triangles)

    return mesh

def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, single_gt_img=False):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    rgb0s = []
    psnr = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        t = time.time()

        rgb, disp, acc, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        # rgb0s.append(extras['rgb0'].cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)
        
        if gt_imgs is not None and render_factor==0:
            if single_gt_img:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs)))
            else:
                p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            psnr.append(p)#print(p)

        if savedir is not None:
            # rgb8_c = to8b(rgb0s[-1]) # save coarse img
            # filename = os.path.join(savedir, '{:03d}_coarse.png'.format(i))
            # imageio.imwrite(filename, rgb8_c)

            rgb8_f = to8b(rgbs[-1]) # save coarse+fine img
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8_f)
            
            ### Need validate code
            rgb_gt = to8b(gt_imgs[i]) # save GT img here
            filename = os.path.join(savedir, '{:03d}_GT.png'.format(i))
            imageio.imwrite(filename, rgb_gt)

            rgb_disp = to8b(disps[-1] / np.max(disps[-1])) # save GT img here
            filename = os.path.join(savedir, '{:03d}_disp.png'.format(i))
            imageio.imwrite(filename, rgb_disp)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    psnr = np.mean(psnr,0)
    print("Mean PSNR of this run is:", psnr)

    return rgbs, disps

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    if args.reduce_embedding==2: # use DNeRF embedding
        embed_fn, input_ch, embedder_obj = get_embedder(args.multires, args.i_embed, args.reduce_embedding, args.epochToMaxFreq) # input_ch.shape=63
    else:
        embed_fn, input_ch, _ = get_embedder(args.multires, args.i_embed, args.reduce_embedding) # input_ch.shape=63

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        if args.reduce_embedding==2: # use DNeRF embedding
            if args.no_DNeRF_viewdir: # no DNeRF embedding for viewdir
                embeddirs_fn, input_ch_views, _ = get_embedder(args.multires_views, args.i_embed)
            else:
                embeddirs_fn, input_ch_views, embedddirs_obj = get_embedder(args.multires_views, args.i_embed, args.reduce_embedding, args.epochToMaxFreq)
        else:
            embeddirs_fn, input_ch_views, _ = get_embedder(args.multires_views, args.i_embed, args.reduce_embedding) # input_ch_views.shape=27
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips, 
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    device = torch.device("cuda")
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        if args.multi_gpu:
            model_fine = torch.nn.DataParallel(model_fine).to(device)
        else:
            model_fine = model_fine.to(device)
        grad_vars += list(model_fine.parameters())

    if args.reduce_embedding==2: # use DNeRF embedding
        network_query_fn = lambda inputs, viewdirs, network_fn, epoch: run_network_DNeRF(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk,
                                                                epoch=epoch, no_DNeRF_viewdir=args.no_DNeRF_viewdir)
    else:
        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.use_ndc_7Scenes:
        pass
    elif args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                i_epoch=-1):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals) # sample in depth
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals)) # sample in diparity disparity = 1/depth

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # find mid points of each interval
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape) # randomly choose in intervals

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3] # Sample 3D point print it out, [batchsize, 64 samples, 3 axis]

    if i_epoch>=0:
        raw = network_query_fn(pts, viewdirs, network_fn, i_epoch)
    else:
        raw = network_query_fn(pts, viewdirs, network_fn) # at line 224: run_network(inputs, viewdirs, network_fn, embed_fn=embed_fn,embeddirs_fn=embeddirs_fn, netchunk=args.netchunk)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        if i_epoch>=0:
            raw = network_query_fn(pts, viewdirs, run_fn, i_epoch)
        else:
            raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def render_test(args, train_dl, val_dl, hwf, start, render_kwargs_test):

    # ### Eval Training set result
    trainsavedir = os.path.join(args.basedir, args.expname, 'evaluate_train_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(trainsavedir, exist_ok=True)
    images_train = []
    poses_train = []
    # views from validation set
    for img, pose in train_dl:
        img_val = img.permute(0,2,3,1) # (1,240,360,3)
        pose_val = torch.zeros(1,4,4)
        pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
        pose_val[0,3,3] = 1.
        images_train.append(img_val)
        poses_train.append(pose_val)

    images_train = torch.cat(images_train, dim=0).numpy()
    poses_train = torch.cat(poses_train, dim=0)
    print('train poses shape', poses_train.shape)

    with torch.no_grad():
        rgbs, disps = render_path(poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir)
    print('Saved train set')
    if args.render_video_train:
        print('Saving trainset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.expname, '{}_trainset_{:06d}_'.format(args.expname, start))
        imageio.mimwrite(moviebase + 'train_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'train_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)

    ### Eval Validation set result
    testsavedir = os.path.join(args.basedir, args.expname, 'evaluate_val_{}_{:06d}'.format('test' if args.render_test else 'path', start))
    os.makedirs(testsavedir, exist_ok=True)
    images_val = []
    poses_val = []
    # views from validation set
    for img, pose in val_dl: 
        img_val = img.permute(0,2,3,1) # (1,240,360,3)
        pose_val = torch.zeros(1,4,4)
        pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
        pose_val[0,3,3] = 1.
        images_val.append(img_val)
        poses_val.append(pose_val)

    images_val = torch.cat(images_val, dim=0).numpy()
    poses_val = torch.cat(poses_val, dim=0)
    print('test poses shape', poses_val.shape)
    with torch.no_grad():
        rgbs, disps = render_path(poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir)
    print('Saved test set')
    if args.render_video_test:
        print('Saving testset as video', rgbs.shape, disps.shape)
        moviebase = os.path.join(args.basedir, args.expname, '{}_test_{:06d}_'.format(args.expname, start))
        imageio.mimwrite(moviebase + 'test_rgb.mp4', to8b(rgbs), fps=15, quality=8)
        imageio.mimwrite(moviebase + 'test_disp.mp4', to8b(disps / np.max(disps)), fps=15, quality=8)
    return

def train_7scene(args, train_dl, val_dl, hwf, i_split, near, far, render_poses=None, render_img=None):

    i_train, i_val, i_test = i_split
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    if args.reduce_embedding==2:
        render_kwargs_train['i_epoch'] = -1
        render_kwargs_test['i_epoch'] = -1

    if args.render_test:
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)
        if args.reduce_embedding==2:
            render_kwargs_test['i_epoch'] = global_step
        render_test(args, train_dl, val_dl, hwf, start, render_kwargs_test)
        return

    # if args.render_pose_only:
    #     # Turn on testing mode
    #     if args.reduce_embedding==2:
    #             render_kwargs_test['i_epoch'] = global_step
    #     with torch.no_grad():
    #         rgbs, disps = render_path(torch.Tensor(render_poses).to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=render_img, single_gt_img=True)
    #     print('Done, saving', rgbs.shape, disps.shape)
    #     moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, global_step))
    #     imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
    #     imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
    #     return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    N_epoch = 4000 + 1 # epoch. Advice: pick 1500~4000 epochs depends on scenes
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)
    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    for i in trange(start, N_epoch):
        time0 = time.time()
        if args.reduce_embedding==2:
            render_kwargs_train['i_epoch'] = i
        # Random from one image for 7 Scenes
        for batch_idx, (target, pose) in enumerate(train_dl):

            target = target[0].permute(1,2,0).to(device) # (240,360,3)
            pose = pose.reshape(3,4).to(device) # reshape to 3x4 rot matrix
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                            torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                        ), -1)
                    if i == start:
                        print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train)

            optimizer.zero_grad()
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][...,-1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss = loss + img_loss0
                psnr0 = mse2psnr(img_loss0)
            loss.backward()
            optimizer.step()

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
            ################################

        dt = time.time()-time0
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0 and i!=0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.N_importance > 0: # have fine sample network
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0: # run thru all validation set
            if args.reduce_embedding==2:
                render_kwargs_test['i_epoch'] = i
            trainsavedir = os.path.join(basedir, expname, 'trainset_{:06d}'.format(i))
            os.makedirs(trainsavedir, exist_ok=True)
            images_train = []
            poses_train = []

            j_skip = 10 # save holdout view render result Trainset/j_skip
            # randomly choose some holdout views from training set
            for batch_idx, (img, pose) in enumerate(train_dl):
                if batch_idx % j_skip != 0:
                    continue
                img_val = img.permute(0,2,3,1) # (1,240,360,3)
                pose_val = torch.zeros(1,4,4)
                pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
                pose_val[0,3,3] = 1.
                images_train.append(img_val)
                poses_train.append(pose_val)
            images_train = torch.cat(images_train, dim=0).numpy()
            poses_train = torch.cat(poses_train, dim=0)
            print('train poses shape', poses_train.shape)
            with torch.no_grad():
                render_path(poses_train.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_train, savedir=trainsavedir)
            print('Saved train set')

            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            images_val = []
            poses_val = []
            # views from validation set
            for img, pose in val_dl:
                img_val = img.permute(0,2,3,1) # (1,240,360,3)
                pose_val = torch.zeros(1,4,4)
                pose_val[0,:3,:4] = pose.reshape(3,4)[:3,:4] # (1,3,4))
                pose_val[0,3,3] = 1.
                images_val.append(img_val)
                poses_val.append(pose_val)

            images_val = torch.cat(images_val, dim=0).numpy()
            poses_val = torch.cat(poses_val, dim=0)
            print('test poses shape', poses_val.shape)
            with torch.no_grad():
                render_path(poses_val.to(device), hwf, args.chunk, render_kwargs_test, gt_imgs=images_val, savedir=testsavedir)
            print('Saved test set')
    
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1