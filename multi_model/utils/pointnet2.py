import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from time import time
import numpy as np
#from .pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from multi_model.utils.pn2_utils.modules import PointNetSAModule, PointnetFPModule
from multi_model.utils.pn2_utils.nn import SharedMLP

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm£»
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dists[dists < 1e-10] = 1e-10
            weight = 1.0 / dists  # [B, N, 3]
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)  # [B, N, 3]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        return new_points

class PointNet2Seg(nn.Module):
    """PointNet++ part segmentation with single-scale grouping

    PointNetSA: PointNet Set Abstraction Layer
    PointNetFP: PointNet Feature Propagation Layer

    Args:
        score_classes (int): the number of grasp score classes
        num_centroids (tuple of int): the numbers of centroids to sample in each set abstraction module
        radius (tuple of float): a tuple of radius to query neighbours in each set abstraction module
        num_neighbours (tuple of int): the numbers of neighbours to query for each centroid
        sa_channels (tuple of tuple of int): the numbers of channels within each set abstraction module
        fp_channels (tuple of tuple of int): the numbers of channels for feature propagation (FP) module
        num_fp_neighbours (tuple of int): the numbers of nearest neighbor used in FP
        seg_channels (tuple of int): the numbers of channels in segmentation mlp
        dropout_prob (float): the probability to dropout input features

    References:
        https://github.com/charlesq34/pointnet2/blob/master/models/pointnet2_part_seg.py

    """
    _SA_MODULE = PointNetSAModule
    _FP_MODULE = PointnetFPModule

    def __init__(self, input_chann = 3, k_score = 1, k_obj = 2, add_channel_flag=False, dropout_prob=0.5):
        super(PointNet2Seg, self).__init__()
        self.k_score = k_score

        num_centroids     = (5120, 1024, 256)
        radius            = (0.02, 0.08, 0.32)
        num_neighbours    = (64, 64, 64)
        sa_channels       = ((128, 128, 256), (256, 256, 512), (512, 512, 1024))
        fp_channels       = ((1024, 1024), (512, 512), (256, 256, 256))
        num_fp_neighbours = (3, 3, 3)
        seg_channels      = (512, 256, 256, 128)

        num_sa_layers = len(num_centroids)
        num_fp_layers = len(fp_channels)

        # Set Abstraction Layers
        feature_channels = input_chann - 3 # 0
        self.sa_modules = nn.ModuleList()
        for ind in range(num_sa_layers):
            sa_module = self._SA_MODULE(in_channels=feature_channels,
                                        mlp_channels=sa_channels[ind],
                                        num_centroids=num_centroids[ind],
                                        radius=radius[ind],
                                        num_neighbours=num_neighbours[ind],
                                        use_xyz=True)
            self.sa_modules.append(sa_module)
            feature_channels = sa_channels[ind][-1]

        inter_channels = [input_chann - 3]# [0]
        inter_channels.extend([x[-1] for x in sa_channels])
        # Feature Propagation Layers
        self.fp_modules = nn.ModuleList()
        feature_channels = inter_channels[-1]
        for ind in range(num_fp_layers):
            fp_module = self._FP_MODULE(in_channels=feature_channels + inter_channels[-2 - ind],
                                        mlp_channels=fp_channels[ind],
                                        num_neighbors=num_fp_neighbours[ind])
            self.fp_modules.append(fp_module)
            feature_channels = fp_channels[ind][-1]

        # MLP
        if not add_channel_flag:
            self.mlp = SharedMLP(feature_channels, seg_channels, ndim=1, dropout_prob=dropout_prob)
        else:
            self.mlp = SharedMLP(feature_channels*3, seg_channels, ndim=1, dropout_prob=dropout_prob)
            
        self.conv_score = nn.Conv1d(seg_channels[-1], self.k_score, 1)
        self.bn_score = nn.BatchNorm1d(self.k_score)
        self.sigmoid = nn.Sigmoid()

    def forward(self, points, add_channel1=None, add_channel2=None):
        B,C,N = points.size()

        xyz = points[:,:3,:]
        feature = points[:,3:6,:]
        
        # save intermediate results
        inter_xyz = [xyz]
        inter_feature = [feature]
        # Set Abstraction Layers
        for sa_module in self.sa_modules:
            xyz, feature = sa_module(xyz, feature)
            inter_xyz.append(xyz)
            inter_feature.append(feature)

        # Feature Propagation Layers
        sparse_xyz = xyz
        sparse_feature = feature
        for fp_ind, fp_module in enumerate(self.fp_modules):
            dense_xyz = inter_xyz[-2 - fp_ind]
            dense_feature = inter_feature[-2 - fp_ind]
            fp_feature = fp_module(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            sparse_xyz = dense_xyz
            sparse_feature = fp_feature

        if add_channel1 is not None and add_channel2 is not None:
            add_channel1 = add_channel1.view(B,1,N).repeat(1,sparse_feature.shape[1],1)
            add_channel2 = add_channel2.view(B,1,N).repeat(1,sparse_feature.shape[1],1)
            sparse_feature = torch.cat((sparse_feature, add_channel1.float(), add_channel2.float()), dim=1)
        # MLP
        x = self.mlp(sparse_feature)
        x_score = self.bn_score(self.conv_score(x))
        x_score = x_score.transpose(2,1).contiguous()
        x_score = self.sigmoid(x_score).view(B, N)

        return sparse_feature, x_score

class PointNet2TwoStage(nn.Module):
    def __init__(self, num_points, input_chann, k_cls, k_reg, k_reg_theta, add_channel_flag=False):
        super(PointNet2TwoStage, self).__init__()
        self.num_points = num_points
        self.k_reg = k_reg
        self.k_cls = k_cls
        self.k_reg_no_anchor = self.k_reg // self.k_cls
        self.k_reg_theta = k_reg_theta

        if not add_channel_flag:
            self.conv = nn.Conv1d(256, 1024, 1)
        else:
            self.conv = nn.Conv1d(256*3, 1024, 1)

        self.bn = nn.BatchNorm1d(1024)

        #self.conv_cls1 = nn.Conv1d(128, 1024, 1)#128128+1024
        self.conv_cls2 = nn.Conv1d(1024, 256, 1)#128+1024
        self.conv_cls3 = nn.Conv1d(256, 128, 1)#128+1024
        self.linear_cls = torch.nn.Linear(128, self.k_cls)
        self.conv_cls4 = nn.Conv1d(128, self.k_cls, 1)
        #self.bn_cls1 = nn.BatchNorm1d(1024)
        self.bn_cls2 = nn.BatchNorm1d(256)
        self.bn_cls3 = nn.BatchNorm1d(128)
        self.bn_cls4 = nn.BatchNorm1d(self.k_cls)

        #self.conv_reg1 = nn.Conv1d(128, 1024, 1)#128+1024
        self.conv_reg2 = nn.Conv1d(1024, 256, 1)#+1024
        self.conv_reg3 = nn.Conv1d(256, 128, 1)#+1024
        self.conv_reg4 = nn.Conv1d(128, self.k_reg, 1)
        #self.bn_reg1 = nn.BatchNorm1d(1024)
        self.bn_reg2 = nn.BatchNorm1d(256)
        self.bn_reg3 = nn.BatchNorm1d(128)
        self.bn_reg4 = nn.BatchNorm1d(self.k_reg)

        #self.conv_reg_theta = nn.Conv1d(int(self.k_reg/7*3), self.k_reg_theta, 1)
        #self.bn_reg_theta = nn.BatchNorm1d(self.k_reg_theta)

        self.mp1 = nn.MaxPool1d(num_points)
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmod = nn.Sigmoid()

    def forward(self, xyz, feature):
        #x = F.avg_pool1d(xyz.float(),self.num_points)
        mp_x = self.mp1(xyz) #[len(true_mask), 128, 1]
        ####mp_x = xyz.view(-1, 128, 1)
        if feature is not None:
            #x[:,:128,:] = x[:,:128,:] + feature.view(feature.shape[0], feature.shape[1],1)
            mp_x = torch.cat((mp_x, feature.view(feature.shape[0], feature.shape[1],1)), dim=1)
            #mp_x = feature.view(feature.shape[0], feature.shape[1],1)
        x = mp_x
        
        x = F.relu(self.bn(self.conv(x)))

        x_cls = F.relu(self.bn_cls2(self.conv_cls2(x)))
        x_cls = F.relu(self.bn_cls3(self.conv_cls3(x_cls)))
        x_cls = self.bn_cls4(self.conv_cls4(x_cls))

        B,C,_ = x_cls.size()
        x_cls = x_cls.view(B,C)

        x_reg = F.relu(self.bn_reg2(self.conv_reg2(x)))
        x_reg = F.relu(self.bn_reg3(self.conv_reg3(x_reg)))
        x_reg = self.bn_reg4(self.conv_reg4(x_reg))

        x_reg = x_reg.view(B,-1,self.k_reg_no_anchor)
        x_reg[:,:,7:] = self.sigmod(x_reg[:,:,7:])
        '''
        x_reg = x_reg.view(B,-1,7)
        x_reg_theta = self.bn_reg_theta(self.conv_reg_theta(x_reg[:,:,3:6].contiguous().view(B,-1,1)))
        x_reg_new = torch.cat([x_reg, x_reg_theta], dim=2)
        x_reg_new = x_reg_new[:,:,[0,1,2,3,4,5,7,6]]
        return x_cls, x_reg_new, mp_x
        '''
        return x_cls, x_reg, mp_x


class PointNet2TwoStageCls(nn.Module):
    def __init__(self, num_points = 2500, input_chann = 3, k_cls = 2, k_reg = 7, k_have_grasp = 2):
        super(PointNet2TwoStageCls, self).__init__()
        self.num_points = num_points
        self.k_reg = k_reg
        self.k_cls = k_cls
        self.k_have_grasp = k_have_grasp

        self.conv = nn.Conv1d(256+1, 1024, 1)#256+1024
        self.bn = nn.BatchNorm1d(1024)

        #self.conv_cls1 = nn.Conv1d(128, 1024, 1)#128128+1024
        self.conv_cls2 = nn.Conv1d(1024, 256, 1)#128+1024
        self.conv_cls3 = nn.Conv1d(256, 128, 1)#128+1024
        self.conv_cls4 = nn.Conv1d(128, self.k_cls, 1)
        #self.bn_cls1 = nn.BatchNorm1d(1024)
        self.bn_cls2 = nn.BatchNorm1d(256)
        self.bn_cls3 = nn.BatchNorm1d(128)
        self.bn_cls4 = nn.BatchNorm1d(self.k_cls)

        #self.conv_reg1 = nn.Conv1d(128, 1024, 1)#128+1024
        self.conv_reg2 = nn.Conv1d(1024, 256, 1)#+1024
        self.conv_reg3 = nn.Conv1d(256, 128, 1)#+1024
        self.conv_reg4 = nn.Conv1d(128, self.k_reg, 1)
        #self.bn_reg1 = nn.BatchNorm1d(1024)
        self.bn_reg2 = nn.BatchNorm1d(256)
        self.bn_reg3 = nn.BatchNorm1d(128)
        self.bn_reg4 = nn.BatchNorm1d(self.k_reg)

        self.conv_cls_grasp2 = nn.Conv1d(1024, 256, 1)#128+1024
        self.conv_cls_grasp3 = nn.Conv1d(256, 128, 1)#128+1024
        self.conv_cls_grasp4 = nn.Conv1d(128, self.k_have_grasp, 1)
        #self.bn_cls1 = nn.BatchNorm1d(1024)
        self.bn_cls_grasp2 = nn.BatchNorm1d(256)
        self.bn_cls_grasp3 = nn.BatchNorm1d(128)
        self.bn_cls_grasp4 = nn.BatchNorm1d(self.k_have_grasp)

        self.sigmod = nn.Sigmoid()

    def forward(self, feature):
        # feature: [B, 257, N]
        x = F.relu(self.bn(self.conv(feature)))

        x_cls = F.relu(self.bn_cls2(self.conv_cls2(x)))
        x_cls = F.relu(self.bn_cls3(self.conv_cls3(x_cls)))
        x_cls = self.bn_cls4(self.conv_cls4(x_cls))
        B,C,N = x_cls.size()
        x_cls = x_cls.transpose(2,1).contiguous().view(B*N,-1)

        x_cls_grasp = F.relu(self.bn_cls_grasp2(self.conv_cls_grasp2(x)))
        x_cls_grasp = F.relu(self.bn_cls_grasp3(self.conv_cls_grasp3(x_cls_grasp)))
        x_cls_grasp = self.bn_cls_grasp4(self.conv_cls_grasp4(x_cls_grasp))
        B,C,N = x_cls_grasp.size()
        x_cls_grasp = x_cls_grasp.transpose(2,1).contiguous().view(B*N,-1)
        x_cls_grasp = self.sigmod(x_cls_grasp)

        x_reg = F.relu(self.bn_reg2(self.conv_reg2(x)))
        x_reg = F.relu(self.bn_reg3(self.conv_reg3(x_reg)))
        x_reg = self.bn_reg4(self.conv_reg4(x_reg))
        x_reg = x_reg.transpose(2,1).contiguous().view(B*N,-1,8)
        return x_cls_grasp, x_cls, x_reg

class PointNet2RefineNoRegion(nn.Module):
    def __init__(self, num_points, input_chann, k_cls, k_reg, add_channel_flag=False):
        super(PointNet2RefineNoRegion, self).__init__()
        self.num_points = num_points
        self.k_reg = k_reg
        self.k_cls = k_cls

        if not add_channel_flag:
            self.conv_formal = nn.Conv1d(256, 1024, 1)
        else:
            self.conv_formal = nn.Conv1d(256*3, 1024, 1)

        # self.conv_formal = nn.Conv1d(384, 1024, 1)
        self.bn_formal = nn.BatchNorm1d(1024)

        # self.conv_formal_cls2 = nn.Conv1d(1024, 128, 1)
        # self.conv_formal_cls3 = nn.Conv1d(128, self.k_cls, 1)
        # self.bn_formal_cls2 = nn.BatchNorm1d(128)
        # self.bn_formal_cls3 = nn.BatchNorm1d(self.k_cls)
        self.conv_formal_cls2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_cls3 = nn.Conv1d(128, self.k_cls, 1)
        self.bn_formal_cls2 = nn.BatchNorm1d(128)
        self.bn_formal_cls3 = nn.BatchNorm1d(self.k_cls)

        self.conv_formal_reg2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_reg3 = nn.Conv1d(128, self.k_reg, 1)
        self.bn_formal_reg2 = nn.BatchNorm1d(128)
        self.bn_formal_reg3 = nn.BatchNorm1d(self.k_reg)

        self.mp1 = nn.MaxPool1d(num_points)
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gripper_feature, group_feature):
        '''
        gripper_feature : [B, region_num, 128]
        group_feature: [B, 128]
        '''
        gripper_feature = self.mp1(gripper_feature) #[B, 128, 1]
        # x: [B, 128, 1]
        x = gripper_feature
        if group_feature is not None:
            # x: [B, 256, 1]
            x = torch.cat((x, group_feature.view(group_feature.shape[0], \
                                group_feature.shape[1],1)), dim=1)
        x = F.relu(self.bn_formal(self.conv_formal(x)))

        # x_cls = F.relu(self.bn_formal_cls2(self.conv_formal_cls2(x)))
        # x_cls = self.bn_formal_cls3(self.conv_formal_cls3(x_cls))
        # x_cls = x_cls.view(x_cls.shape[0], x_cls.shape[1])

        x_cls = F.relu(self.bn_formal_cls2(self.conv_formal_cls2(x)))
        x_cls = self.bn_formal_cls3(self.conv_formal_cls3(x_cls))
        x_cls = x_cls.view(x_cls.shape[0], x_cls.shape[1])

        x_reg = F.relu(self.bn_formal_reg2(self.conv_formal_reg2(x)))
        x_reg = self.bn_formal_reg3(self.conv_formal_reg3(x_reg))
        x_reg = x_reg.view(x_reg.shape[0], x_reg.shape[1])
        #x_reg[:,-1] = self.sigmoid(x_reg[:,-1])
        
        return x_cls, x_reg
        # return x_reg

class PointNet2Refine(nn.Module):
    def __init__(self, num_points = 2500, input_chann = 3, k_cls = 2, k_reg = 8):
        super(PointNet2Refine, self).__init__()
        self.num_points = num_points
        self.k_reg = k_reg
        self.k_cls = k_cls

        self.conv_formal = nn.Conv1d(384, 1024, 1)
        self.bn_formal = nn.BatchNorm1d(1024)

        # self.conv_formal_cls2 = nn.Conv1d(1024, 128, 1)
        # self.conv_formal_cls3 = nn.Conv1d(128, self.k_cls, 1)
        # self.bn_formal_cls2 = nn.BatchNorm1d(128)
        # self.bn_formal_cls3 = nn.BatchNorm1d(self.k_cls)
        self.conv_formal_cls2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_cls3 = nn.Conv1d(128, self.k_cls, 1)
        self.bn_formal_cls2 = nn.BatchNorm1d(128)
        self.bn_formal_cls3 = nn.BatchNorm1d(self.k_cls)

        self.conv_formal_reg2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_reg3 = nn.Conv1d(128, self.k_reg, 1)
        self.bn_formal_reg2 = nn.BatchNorm1d(128)
        self.bn_formal_reg3 = nn.BatchNorm1d(self.k_reg)

        self.mp1 = nn.MaxPool1d(num_points)
        self.ap = torch.nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gripper_feature, group_feature):
        '''
        gripper_feature : [B, region_num, 128]
        group_feature: [B, 128]
        '''
        gripper_feature = self.mp1(gripper_feature) #[B, 128, 1]
        # x: [B, 128, 1]
        x = gripper_feature
        if group_feature is not None:
            # x: [B, 256, 1]
            x = torch.cat((x, group_feature.view(group_feature.shape[0], \
                                group_feature.shape[1],1)), dim=1)
        x = F.relu(self.bn_formal(self.conv_formal(x)))

        # x_cls = F.relu(self.bn_formal_cls2(self.conv_formal_cls2(x)))
        # x_cls = self.bn_formal_cls3(self.conv_formal_cls3(x_cls))
        # x_cls = x_cls.view(x_cls.shape[0], x_cls.shape[1])

        x_cls = F.relu(self.bn_formal_cls2(self.conv_formal_cls2(x)))
        x_cls = self.bn_formal_cls3(self.conv_formal_cls3(x_cls))
        x_cls = x_cls.view(x_cls.shape[0], x_cls.shape[1])

        x_reg = F.relu(self.bn_formal_reg2(self.conv_formal_reg2(x)))
        x_reg = self.bn_formal_reg3(self.conv_formal_reg3(x_reg))
        x_reg = x_reg.view(x_reg.shape[0], x_reg.shape[1])
        #x_reg[:,-1] = self.sigmoid(x_reg[:,-1])
        
        return x_cls, x_reg
        # return x_reg

class PointNet2RefineCls(nn.Module):
    def __init__(self, num_points = 2500, input_chann = 3, k_cls = 2, k_reg = 1):
        super(PointNet2RefineCls, self).__init__()
        self.num_points = num_points
        self.k_reg = k_reg
        self.k_cls = k_cls

        self.conv_formal = nn.Conv1d(256, 1024, 1)
        self.bn_formal = nn.BatchNorm1d(1024)

        self.conv_formal_cls2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_cls3 = nn.Conv1d(128, self.k_cls, 1)
        self.bn_formal_cls2 = nn.BatchNorm1d(128)
        self.bn_formal_cls3 = nn.BatchNorm1d(self.k_cls)

        self.conv_formal_reg2 = nn.Conv1d(1024, 128, 1)
        self.conv_formal_reg3 = nn.Conv1d(128, self.k_reg, 1)
        self.bn_formal_reg2 = nn.BatchNorm1d(128)
        self.bn_formal_reg3 = nn.BatchNorm1d(self.k_reg)

        self.mp1 = nn.MaxPool1d(num_points)
        self.ap = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, gripper_feature, group_feature):
        '''
        gripper_feature : [B, region_num, 128]
        group_feature: [B, 128]
        '''
        gripper_feature = self.mp1(gripper_feature) #[B, 128, 1]
        # x: [B, 128, 1]
        x = gripper_feature
        if group_feature is not None:
            # x: [B, 256, 1]
            x = torch.cat((x, group_feature.view(group_feature.shape[0], \
                                group_feature.shape[1],1)), dim=1)
        x = F.relu(self.bn_formal(self.conv_formal(x)))

        x_cls = F.relu(self.bn_formal_cls2(self.conv_formal_cls2(x)))
        x_cls = self.bn_formal_cls3(self.conv_formal_cls3(x_cls))
        x_cls = x_cls.view(x_cls.shape[0], x_cls.shape[1])

        x_reg = F.relu(self.bn_formal_reg2(self.conv_formal_reg2(x)))
        x_reg = self.bn_formal_reg3(self.conv_formal_reg3(x_reg))
        x_reg = x_reg.view(x_reg.shape[0], x_reg.shape[1])
        
        return x_cls, x_reg

if __name__ == '__main__':
    pass
