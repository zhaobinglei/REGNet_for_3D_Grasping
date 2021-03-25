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

if __name__ == '__main__':
    pass
