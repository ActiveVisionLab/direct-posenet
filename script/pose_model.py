import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import numpy as np
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# for get_error_in_q
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
import math
import time

def preprocess_data(inputs, device):
    # normalize inputs according to https://pytorch.org/hub/pytorch_vision_mobilenet_v2/
    mean = torch.Tensor([0.485, 0.456, 0.406]).to(device) # per channel subtraction
    std = torch.Tensor([0.229, 0.224, 0.225]).to(device) # per channel division
    inputs = (inputs - mean[None,:,None,None])/std[None,:,None,None]
    return inputs

def filter_hook(m, g_in, g_out):
    g_filtered = []
    for g in g_in:
        g = g.clone()
        g[g != g] = 0
        g_filtered.append(g)
    return tuple(g_filtered)

# From Mapnet
def quaternion_angular_error(q1, q2):
  """
  angular error between two quaternions
  :param q1: (4, )
  :param q2: (4, )
  :return:
  """
  d = abs(np.dot(q1, q2))
  d = min(1.0, max(-1.0, d))
  theta = 2 * np.arccos(d) * 180 / np.pi
  return theta

# loss functions
t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt) # Frobenius norm
q_criterion = quaternion_angular_error

def get_error_in_q(args, dl, model, sample_size, device, batch_size=1):
    ''' Convert Rotation matrix to quaternion, then calculate the location errors. original from PoseNet Paper '''
    model.eval()
    use_SVD=True # Turn on for Direct-PN and Direct-PN+U reported result, despite it makes minuscule differences
    time_spent = []
    results = np.zeros((sample_size, 2))
    i = 0
    for data, pose in dl:
        data = data.to(device) # input
        pose = pose.reshape((batch_size,3,4)).numpy() # label

        if args.preprocess_ImgNet:
            data = preprocess_data(data, device)

        if use_SVD:
            # using SVD to make sure predict rotation is normalized rotation matrix
            with torch.no_grad():
                predict_pose = model(data)
                R_torch = predict_pose.reshape((batch_size, 3, 4))[:,:3,:3] # debug
                predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()

            R = predict_pose[:,:3,:3]
            res = R@np.linalg.inv(R)
            # print('R@np.linalg.inv(R):', res)

            u,s,v=torch.svd(R_torch)
            Rs = torch.matmul(u, v.transpose(-2,-1))
            predict_pose[:,:3,:3] = Rs[:,:3,:3].cpu().numpy()
        else:
            start_time = time.time()
            # inference NN
            with torch.no_grad():
                predict_pose = model(data)
                predict_pose = predict_pose.reshape((batch_size, 3, 4)).cpu().numpy()
            time_spent.append(time.time() - start_time)

        pose_q = tfg_transformation.quaternion.from_rotation_matrix(pose[:,:3,:3]) # gnd truth in quaternion
        pose_x = pose[:, :3, 3] # gnd truth position
        predicted_q = tfg_transformation.quaternion.from_rotation_matrix(predict_pose[:, :3, :3]) # predict in quaternion
        predicted_x = predict_pose[:, :3, 3] # predict position

        pose_q = tf.squeeze(pose_q) 
        pose_x = tf.squeeze(pose_x) 
        predicted_q = tf.squeeze(predicted_q) 
        predicted_x = tf.squeeze(predicted_x)

        #Compute Individual Sample Error 
        q1 = pose_q / tf.linalg.norm(pose_q)
        q2 = predicted_q / tf.linalg.norm(predicted_q)
        d = tf.math.abs(tf.reduce_sum(tf.math.multiply(q1,q2))) 
        d = tf.clip_by_value(d, -1., 1.) # acos can only input [-1~1]

        theta = 2 * tf.math.acos(d) * 180/math.pi
        error_x = tf.linalg.norm(pose_x-predicted_x)
        results[i,:] = [error_x, theta]
        #print ('Iteration: {} Error XYZ (m): {} Error Q (degrees): {}'.format(i, error_x, theta))
        i += 1
    median_result = np.median(results,axis=0)
    mean_result = np.mean(results,axis=0)
    print ('Median error {}m and {} degrees.'.format(median_result[0], median_result[1]))
    print ('Mean error {}m and {} degrees.'.format(mean_result[0], mean_result[1]))
    #print ('Avg execution time (sec): {:.3f}'.format(np.mean(time_spent)))

    # num_translation_less_5cm = np.asarray(np.where(results[:,0]<0.05))[0]
    # num_rotation_less_5 = np.asarray(np.where(results[:,1]<5))[0]
    # print ('translation error less than 5cm {}/{}.'.format(num_translation_less_5cm.shape[0], results.shape[0]))
    # print ('rotation error less than 5 degree {}/{}.'.format(num_rotation_less_5.shape[0], results.shape[0]))
    # print ('results:', results)
    if 0:
      filename='Direct-PN+U_' + args.datadir.split('/')[-1] + '_result.txt'
      np.savetxt(filename, predict_pose)

# PoseNet (SE(3)) w/ mobilev2 backbone
class PoseNetV2(nn.Module):
    def __init__(self, feat_dim=12):
        super(PoseNetV2, self).__init__()
        self.backbone_net = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = self.backbone_net.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_pose = nn.Linear(1280, feat_dim)

    def forward(self, Input):
        x = self.feature_extractor(Input)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        predict = self.fc_pose(x)
        return predict

# PoseNet (SE(3)) w/ resnet34 backnone. We found dropout layer is unnecessary, so we set droprate as 0 in reported results.
class PoseNet_res34(nn.Module):
    def __init__(self, droprate=0.5, pretrained=True,
        feat_dim=2048):
        super(PoseNet_res34, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)
        self.fc_pose = nn.Linear(feat_dim, 12)

        # initialize
        if pretrained:
          init_modules = [self.feature_extractor.fc]
        else:
          init_modules = self.modules()

        for m in init_modules:
          if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
              nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.droprate > 0:
          x = F.dropout(x, p=self.droprate)
        predict = self.fc_pose(x)
        return predict


# from MapNet paper CVPR 2018
class PoseNet(nn.Module):
    def __init__(self, feature_extractor, droprate=0.5, pretrained=True,
        feat_dim=2048, filter_nans=False):
        super(PoseNet, self).__init__()
        self.droprate = droprate

        # replace the last FC layer in feature extractor
        self.feature_extractor = models.resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        fe_out_planes = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(fe_out_planes, feat_dim)

        self.fc_xyz  = nn.Linear(feat_dim, 3)
        self.fc_wpqr = nn.Linear(feat_dim, 3)
        if filter_nans:
          self.fc_wpqr.register_backward_hook(hook=filter_hook)
        # initialize
        if pretrained:
          init_modules = [self.feature_extractor.fc, self.fc_xyz, self.fc_wpqr]
        else:
          init_modules = self.modules()

        for m in init_modules:
          if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
              nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = F.relu(x)
        if self.droprate > 0:
          x = F.dropout(x, p=self.droprate)

        xyz  = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)
        return torch.cat((xyz, wpqr), 1)

class MapNet(nn.Module):
    """
    Implements the MapNet model (green block in Fig. 2 of paper)
    """
    def __init__(self, mapnet):
        """
        :param mapnet: the MapNet (two CNN blocks inside the green block in Fig. 2
        of paper). Not to be confused with MapNet, the model!
        """
        super(MapNet, self).__init__()
        self.mapnet = mapnet

    def forward(self, x):
        """
        :param x: image blob (N x T x C x H x W)
        :return: pose outputs
         (N x T x 6)
        """
        s = x.size()
        x = x.view(-1, *s[2:])
        poses = self.mapnet(x)
        poses = poses.view(s[0], s[1], -1)
        return poses

def eval_on_epoch(args, dl, model, optimizer, loss_func, device):
    model.eval()
    val_loss_epoch = []
    for data, pose in dl:
        inputs = data.to(device)
        labels = pose.to(device)
        if args.preprocess_ImgNet:
            inputs = preprocess_data(inputs, device)
        predict = model(inputs)
        loss = loss_func(predict, labels)
        val_loss_epoch.append(loss.item())
    val_loss_epoch_mean = np.mean(val_loss_epoch)
    return val_loss_epoch_mean


def train_on_epoch(args, dl, model, optimizer, loss_func, device):
    model.train()
    train_loss_epoch = []
    for data, pose in dl:
        inputs = data.to(device) # (N, Ch, H, W) ~ (4,3,200,200), 7scenes [4, 3, 256, 341] wierd shape...
        labels = pose.to(device)
        if args.preprocess_ImgNet:
            inputs = preprocess_data(inputs, device)

        predict = model(inputs)
        loss = loss_func(predict, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss_epoch_mean = np.mean(train_loss_epoch)
    return train_loss_epoch_mean

def train_posenet(args, train_dl, val_dl, model, epochs, optimizer, loss_func, scheduler, device, early_stopping):
    writer = SummaryWriter()
    model_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in tqdm(range(epochs), desc='epochs'):
        
        # train 1 epoch
        train_loss = train_on_epoch(args, train_dl, model, optimizer, loss_func, device)
        writer.add_scalar("Loss/train", train_loss, epoch)
        
        # validate every epoch
        val_loss = eval_on_epoch(args, val_dl, model, optimizer, loss_func, device)
        writer.add_scalar("Loss/val", val_loss, epoch)
        
        # reduce LR on plateau
        scheduler.step(val_loss)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)

        # logging
        tqdm.write('At epoch {0:6d} : train loss: {1:.4f}, val loss: {2:.4f}'.format(epoch, train_loss, val_loss))
                
        # check wether to early stop
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')

        if epoch%50 == 0:
            get_error_in_q(args, val_dl, model, len(val_dl.dataset), device, batch_size=1)


    writer.flush()
