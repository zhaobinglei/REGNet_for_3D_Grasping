import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from multi_model.utils.pointnet2 import PointNet2Seg

class ScoreNetwork(nn.Module):
    def __init__(self, training=True, k_obj=2):
        super(ScoreNetwork, self).__init__()
        self.is_training = training
        self.k_obj = k_obj
        self.extrat_featurePN2 = PointNet2Seg(input_chann=6, k_score=1, k_obj=self.k_obj)
        self.criterion_cls = nn.NLLLoss(reduction='mean')
        self.criterion_reg = nn.MSELoss(reduction='mean')

    def compute_loss(self, pscore, tscore):#, plabel, tlabel, predict_label):
        '''
          Input:  
            pscore        : [B,N] 
            tscore        : [B,N] 
            plabel        : [BxN, k_obj]
            tlabel        : [B,N]
            predict_label : [BxN]
        '''
        loss_point_score = self.criterion_reg(pscore, tscore.float())
        loss = loss_point_score 
        return loss

    def forward(self, pc, pc_score=None, pc_label=None):
        '''
         Input:
          pc              :[B,A,6]
          pc_score        :[B,A]
          pc_label        :[B,A]
         Output:
          all_feature     :[B,A,Feature(128)])
          output_score    :[B,A]
          loss
        '''
        B, N, _ = pc.shape
        #import time 
        #st = time.time()
        #@ all_feature[B, N, 128], output_score[B, N]
        all_feature, output_score = self.extrat_featurePN2(pc[:,:,:6].permute(0,2,1)) 
        #end = time.time()
        all_feature = all_feature.transpose(2,1) 
        loss = None
        if self.is_training and pc_score is not None:# and pc_label is not None:
            loss = self.compute_loss(output_score, pc_score)#, output_label, pc_label, predict_label)

        return all_feature, output_score, loss

if __name__ == '__main__':
    pass
