import numpy as np
import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
#from multi_model.utils.pointnet_test import PointNetPartFeature
from multi_model.utils.pointnet2 import PointNet2TwoStage, PointNet2Refine, PointNet2RefineNoRegion

class GripperRegionNetwork(nn.Module):
    def __init__(self, training, group_num, gripper_num, grasp_score_threshold, radius, reg_channel):
        super(GripperRegionNetwork, self).__init__()
        self.group_number = group_num
        self.templates = _enumerate_templates()
        self.anchor_number = self.templates.shape[1]*self.templates.shape[2]
        self.gripper_number = gripper_num
        self.grasp_score_thre = grasp_score_threshold
        self.is_training_refine = training
        self.radius = radius
        self.reg_channel = reg_channel

        self.extrat_feature_region = PointNet2TwoStage(num_points=group_num, input_chann=6, k_cls=self.anchor_number,\
                                            k_reg=self.reg_channel*self.anchor_number, k_reg_theta=self.anchor_number)
        self.extrat_feature_refine = PointNet2Refine(num_points=gripper_num, input_chann=6, k_cls=2, k_reg=self.reg_channel)

        self.criterion_cos = nn.CosineEmbeddingLoss(reduction='mean')
        self.criterion_cls = torch.nn.CrossEntropyLoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')

    def _enumerate_anchors(self, centers):
        '''
          Enumerate anchors.
          Input:
            centers: [B*num of centers, 3] -> x, y, z
            self.templates :[1,8,1,4] -> 8:rxryrz_num 1:theta_num 4:rxryrztheta
          Return:
            t_anchors: [B*num of centers, 8, 7] -> the number of anchors is 8
                                                     7 means (x, y, z, rx, ry, rz, theta)
        '''
        if centers.cuda:
            self.templates = self.templates.cuda()
        t_center = centers.view(centers.shape[0],1,1,-1).repeat(1,self.templates.shape[1],self.templates.shape[2],1)
        t_anchors = torch.cat( [t_center, self.templates.float().repeat(centers.shape[0],1,1,1)], dim=-1).view(-1, self.templates.shape[1]*self.templates.shape[2], 7)
        return t_anchors

    def compute_loss(self, first_grasp, anchors, first_cls, ground):
        '''
          Input:  
            first_grasp : [B*center_num, num_anchor, 8]  the regression grasps after the first regrssion
            anchors     : [B*center_num, num_anchor, 7]  the anchor grasps
            first_cls   : [B*center_num, num_anchor]     the classification score of grasps after the first regrssion
            ground      : [B, center_num, 8]               ground truth. 8 means the (p,r,theta,score) of a grasp.
          Return:
            next_grasp  : [len(select_center_index), 7]
            loss_tuple, correct_tuple
            next_gt     : [len(select_center_index), 7]
        '''
        #### gmask: [len(gmask)]  true mask index of centers which has a corresponding grasp 
        BmulN_C = len(first_grasp)
        # B, N_C = ground.shape[0], ground.shape[1]
        if ground is not None:
            gmask = torch.nonzero(ground.view(-1,ground.shape[2])[:,-1] != -1).view(-1)
            print(BmulN_C, "centers has", len(gmask), "grasps" )              
        else:
            gmask = torch.arange(0, BmulN_C)
        if first_grasp.cuda:
            gmask = gmask.cuda()

        anchors = anchors[gmask, :, :]
        ##### tt: [num_anchor*len(true_mask), 7]   the anchor grasps
        tt = anchors.clone().detach().transpose(1,0).contiguous().view(-1,7)
        first_grasp, first_cls = first_grasp[gmask], first_cls[gmask]

        first_grasp = first_grasp.transpose(1,0).contiguous().view(-1, first_grasp.shape[2])
        num_anchor = self.templates.shape[1]*self.templates.shape[2]
        _, predict_8 = torch.max(first_cls.transpose(1,0).contiguous(), dim=0)
        # print(predict_8)
        final_mask = predict_8.clone().view(-1)
        for i in range(len(final_mask)):
            final_mask[i] = final_mask[i] * len(final_mask) + i
        first_grasp_pre, tt_pre = first_grasp[final_mask], tt[final_mask]
        sum_r_pre = torch.sqrt(torch.sum(torch.mul(first_grasp_pre[:,3:6]+tt_pre[:,3:6], first_grasp_pre[:,3:6]+tt_pre[:,3:6]), dim=1).add_(1e-12) ).view(-1,1)
        first_grasp_center_pre = first_grasp_pre[:, :3]*self.radius + tt_pre[:,:3]
        first_grasp_r_pre = torch.div(first_grasp_pre[:,3:6]+tt_pre[:,3:6], sum_r_pre)
        first_grasp_angle_pre = np.pi * (first_grasp_pre[:,6:7]+tt_pre[:,6:7])
        first_grasp_score_pre = first_grasp_pre[:,7:]
        # (sinx, cosx)
        #first_grasp_angle_pre = torch.atan2(first_grasp_pre[:,-3].view(-1), first_grasp_pre[:,-2].view(-2)).view(-1,1)
        next_grasp = torch.cat((first_grasp_center_pre, first_grasp_r_pre, \
                                    first_grasp_angle_pre, first_grasp_score_pre), dim=-1)
        
        loss_tuple = (None, None)
        correct_tuple, next_gt, tt_gt = (None, None, None, None), None, None

        if ground is not None:

            repeat_ground = ground[:,:,:7].contiguous().view(-1, 7)[gmask, :].repeat(self.templates.shape[1]*self.templates.shape[2],1)
            repeat_ground_truth = ground[:,:,7:].contiguous().view(-1, ground.shape[2]-7)[gmask, :].repeat(self.templates.shape[1]*self.templates.shape[2],1)
            ## r_sim: [num_anchor, len(gmask)]
            r_sim = compute_cos_sim(tt[:,3:6], repeat_ground[:,3:6]).view(-1).view(num_anchor, -1)

            ## ground_8: [len(gmask)]
            sim = r_sim.clone().transpose(1,0)
            sort_cls, sort_index = torch.sort(sim, dim=1, descending=False)
            ground_8 = sort_index[:,0].view(-1)
            print(ground_8)
            iou_nonzero = ground_8.clone()
            for i in range(len(iou_nonzero)):
                iou_nonzero[i] = iou_nonzero[i] * len(iou_nonzero) + i
            
            len_ground_anchor, num_0, num_t = np.zeros([num_anchor]), 0, 0
            for i in range(num_anchor):
                len_ground_anchor[i] = (ground_8==i).sum()
                if len_ground_anchor[i] == 0:
                    num_0 += 1
            for i in range(num_anchor):
                print(i, "num:", (predict_8==i).sum()) 
            min_len_ground_anchor = len_ground_anchor.min()
            if min_len_ground_anchor == 0:
                min_len_ground_anchor = 1
            ground_anchor_index = torch.zeros([num_anchor-num_0, int(min_len_ground_anchor)])
            for i in range(num_anchor):
                cur_anchor_index = torch.nonzero(ground_8==i).view(-1)
                if len(cur_anchor_index) == 0:
                    continue
                ground_anchor_index[num_t] = cur_anchor_index[np.random.choice(len(cur_anchor_index), \
                                                          int(min_len_ground_anchor), replace=False)]
                num_t += 1
                
            ground_anchor_index = ground_anchor_index.view(-1).long()
            if ground.is_cuda:
                ground_anchor_index = ground_anchor_index.cuda()
            loss_class = self.criterion_cls(first_cls[ground_anchor_index], ground_8[ground_anchor_index].long())
            print("regression stage 1 class loss:", loss_class)
            
            Tcls = ( ground_8 == predict_8).sum().float()
            Fcls = (ground_8 != predict_8).sum().float()
            correct_tuple = (Tcls, Fcls)
            acc = Tcls / (Tcls + Fcls)
            print(Tcls, Fcls, "acc1:", acc)
            

            first_grasp_gt, tt_gt  = first_grasp[iou_nonzero], tt[iou_nonzero]
            sum_r_gt               = torch.sqrt(torch.sum(torch.mul(first_grasp_gt[:,3:6]+tt_gt[:,3:6], \
                                                first_grasp_gt[:,3:6]+tt_gt[:,3:6]), dim=1).add_(1e-12) ).view(-1,1)
            first_grasp_center_gt  = first_grasp_gt[:,:3]*self.radius + tt_gt[:,:3]
            first_grasp_r_gt       = torch.div(first_grasp_gt[:,3:6]+tt_gt[:,3:6], sum_r_gt)
            first_grasp_delta_r_gt = torch.mul(first_grasp_gt[:,3:6], sum_r_gt)
            first_grasp_angle_gt   = np.pi * (first_grasp_gt[:,6:7]+tt_gt[:,6:7])
            first_grasp_score_gt   = first_grasp_gt[:,7:]
            # (sinx, cosx)
            #first_grasp_angle_gt = torch.atan2(first_grasp_gt[:,-1].view(-1,1), first_grasp_gt[:,-2].view(-1,1)).view(-1,1)

            ground_gt = repeat_ground[iou_nonzero]                 # same as repeat_ground[final_mask]
            ground_score_gt = repeat_ground_truth[iou_nonzero]  # same as repeat_ground_truth[final_mask]
        
            loss_first1_gt  = F.smooth_l1_loss(first_grasp_gt[:,:3],  (ground_gt[:,:3]-tt_gt[:,:3]) / self.radius, reduction='mean')
            loss_first2_gt  = F.smooth_l1_loss(first_grasp_delta_r_gt, ground_gt[:,3:6]-tt_gt[:,3:6], reduction='mean')
            loss_first3_gt  = F.smooth_l1_loss(first_grasp_gt[:,6:7], (ground_gt[:,6:7]-tt_gt[:,6:7]) / np.pi, reduction='mean')
            loss_first4_gt  = F.smooth_l1_loss(first_grasp_gt[:,7:],   ground_score_gt, reduction='mean')
            print("regress loss of stage2", loss_first1_gt.data, loss_first2_gt.data, loss_first3_gt.data, loss_first4_gt.data)

            tensor_y_gt = torch.ones(len(iou_nonzero), 1)
            loss_center_gt  = F.smooth_l1_loss  (first_grasp_center_gt, ground_gt[:,:3], reduction='mean').data
            loss_cos_r_gt   = self.criterion_cos(first_grasp_r_gt, ground_gt[:,3:6], tensor_y_gt.cuda()).data
            loss_theta_gt   = F.smooth_l1_loss  (first_grasp_angle_gt, ground_gt[:,6:7], reduction='mean').data
            loss_score_gt   = loss_first4_gt.data
            print("under gt class loss", loss_center_gt, loss_cos_r_gt, loss_theta_gt, loss_score_gt)

            tensor_y_pre = torch.ones(len(final_mask), 1)
            loss_center_pre = F.smooth_l1_loss  ( first_grasp_center_pre, ground_gt[:,:3], reduction='mean').data
            loss_cos_r_pre  = self.criterion_cos( first_grasp_r_pre, ground_gt[:,3:6], tensor_y_pre.cuda()).data
            loss_theta_pre  = F.smooth_l1_loss  ( first_grasp_angle_pre, ground_gt[:,6:7], reduction='mean').data
            loss_score_pre  = F.smooth_l1_loss  ( first_grasp_score_pre, ground_score_gt, reduction='mean').data
            print("under pre class loss", loss_center_pre, loss_cos_r_pre, loss_theta_pre, loss_score_pre)
            
            #next_gt: [len(gmask), 10]
            next_gt = torch.cat((ground_gt, ground_score_gt), dim=1)

            loss = loss_first1_gt*10 + loss_first2_gt*5 + loss_first3_gt + loss_first4_gt + loss_class
            loss_tuple = (loss, loss_class.data, loss_first1_gt.data, loss_first2_gt.data, loss_first3_gt.data, \
                            loss_first4_gt.data, loss_center_pre, loss_cos_r_pre, loss_theta_pre, loss_score_pre, )

        return next_grasp, loss_tuple, correct_tuple, next_gt, tt_gt, gmask

    def compute_loss_refine(self, next_grasp, next_x_cls, next_x_reg, next_gt):
        '''
          Input:
            next_grasp      :[len(gripper_mask),8] regressed grasp from the stage1
            next_x_cls      :[len(gripper_mask),2]
            next_x_reg      :[len(gripper_mask),8] delta grasp from the stage2 (loss)
            next_gt         :[len(gripper_mask),8] ground truth
          Return:
            final_grasp_select       : [len(class_select), 8] 
            select_grasp_class_stage2: [len(class_select), 8]
            class_select             : [len(class_select)]
            loss_stage2              : tuple
            correct_stage2_tuple     : tuple
        '''
        print("Refine Module init number:", next_grasp.shape[0])
        final_grasp = next_grasp.clone()
        final_grasp[:,:3] = final_grasp[:,:3] + next_x_reg[:,:3] * self.radius
        final_grasp[:,3:] = final_grasp[:,3:] + next_x_reg[:,3:] 
        
        # next_x_cls[:,1] += 1
        predit_formal = torch.max(next_x_cls, dim=-1)[1]
        print("########################################")
        print("predict class 0:", torch.sum((predit_formal==0)), "; predict class 1:", torch.sum((predit_formal==1)))
        class_select  = torch.nonzero(predit_formal==1).view(-1)
        score_mask    = (predit_formal==1) & (final_grasp[:,7] > self.grasp_score_thre)
        score_select  = torch.nonzero(score_mask).view(-1)

        select_grasp_class  = final_grasp[class_select].data
        select_grasp_score  = final_grasp[score_select].data
        select_grasp_class_stage2 = next_grasp[class_select].data

        print("final grasp: {}".format(len(select_grasp_class)))
        print("final >{} score grasp: {}".format(self.grasp_score_thre, len(score_select)))
        print("########################################")
        loss_refine_tuple, correct_refine_tuple = (None, None), (None, None, None, None)

        if next_gt is not None:
            gt_class = torch.zeros((len(next_gt)))
            if next_grasp.is_cuda:
                gt_class = gt_class.cuda()
            
            center_dist = (next_grasp[:,:3] - next_gt[:,:3]) 
            center_dist_mask = (torch.sqrt(torch.mul(center_dist[:,0],center_dist[:,0])+torch.mul(center_dist[:,1],center_dist[:,1])\
                                                +torch.mul(center_dist[:,2],center_dist[:,2])) < 0.025).view(-1) 
                            
            r_sim = compute_cos_sim(next_grasp[:,3:6], next_gt[:,3:6]).view(-1)
            r_sim_mask = (r_sim < 0.234).view(-1) # cos60 = 0.5 #0.234
            theta_sim = torch.abs(next_grasp[:,6] - next_gt[:,6]) 
            theta_sim_mask = (theta_sim < 1.047).view(-1) # 1.047 = 60/180*np.pi

            class_mask = (center_dist_mask & r_sim_mask & theta_sim_mask)
            gt_class[class_mask] = 1
            gt_class_1 = torch.nonzero(gt_class == 1).view(-1)
            gt_class_0 = torch.nonzero(gt_class == 0).view(-1)

            num_0, num_1 = len(gt_class_0), len(gt_class_1)
            print("class 0:", num_0, "; class 1:", num_1)
            num = min(num_0, num_1)
            loss = torch.tensor((0), dtype=torch.float)
            loss_class, loss_grasp_center, loss_grasp_r, loss_grasp_theta, loss_grasp_score = loss.clone(), loss.clone(), loss.clone(), loss.clone(), loss.clone()

            loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2 = loss.clone(), loss.clone(), loss.clone(), loss.clone()
            loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre = loss.clone(), loss.clone(), loss.clone(), loss.clone()
            loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score = loss.clone(), loss.clone(), loss.clone(), loss.clone()

            if next_x_cls.is_cuda:
                loss, loss_class = loss.cuda(), loss_class.cuda()
                loss_grasp_center, loss_grasp_r, loss_grasp_theta, loss_grasp_score = loss_grasp_center.cuda(), loss_grasp_r.cuda(), loss_grasp_theta.cuda(), loss_grasp_score.cuda()
                
                loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2 = loss_center_pre_stage2.cuda(), loss_r_cos_pre_stage2.cuda(), loss_theta_pre_stage2.cuda(), loss_score_pre_stage2.cuda()
                loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre = loss_center_pre.cuda(), loss_r_cos_pre.cuda(), loss_theta_pre.cuda(), loss_score_pre.cuda()
                loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score = loss_center_pre_score.cuda(), loss_r_cos_pre_score.cuda(), loss_theta_pre_score.cuda(), loss_score_pre_score.cuda()
                
            if num > 0:
                index_0 = gt_class_0[np.random.choice(num_0, num, replace=False)].view(-1)
                index_1 = gt_class_1[np.random.choice(num_1, num, replace=False)].view(-1)
                index = torch.cat((index_0, index_1), dim=-1)
                loss_class = self.criterion_cls(next_x_cls.view(-1,2)[index], gt_class.view(-1)[index].long())

                loss_grasp_center = F.smooth_l1_loss(next_x_reg[gt_class_1,:3], (next_gt[gt_class_1,:3]-next_grasp[gt_class_1,:3]) / self.radius, reduction='mean')
                loss_grasp_r      = F.smooth_l1_loss(next_x_reg[gt_class_1,3:6], (next_gt[gt_class_1,3:6]-next_grasp[gt_class_1,3:6]) , reduction='mean')
                loss_grasp_theta  = F.smooth_l1_loss(next_x_reg[gt_class_1,6], (next_gt[gt_class_1,6]-next_grasp[gt_class_1,6]) , reduction='mean')
                loss_grasp_score  = F.smooth_l1_loss(next_x_reg[gt_class_1,7:], (next_gt[gt_class_1,7:]-next_grasp[gt_class_1,7:]) , reduction='mean')
                loss              = loss_class + loss_grasp_center + loss_grasp_r + loss_grasp_theta + loss_grasp_score

            if len(class_select) > 0:
                tensor_y = torch.ones(len(class_select), 1)
                if next_x_cls.is_cuda:
                    tensor_y = tensor_y.cuda()
                loss_center_pre        = F.smooth_l1_loss(select_grasp_class[:,:3], next_gt[class_select,:3], reduction='mean').data
                loss_center_pre_score  = F.smooth_l1_loss(select_grasp_score[:,:3], next_gt[score_select,:3], reduction='mean').data
                loss_center_pre_stage2 = F.smooth_l1_loss(select_grasp_class_stage2[:,:3], next_gt[class_select,:3], reduction='mean').data

                loss_r_cos_pre        = self.criterion_cos(select_grasp_class[:,3:6], next_gt[class_select,3:6],tensor_y).data
                loss_r_cos_pre_score  = self.criterion_cos(select_grasp_score[:,3:6], next_gt[score_select,3:6],tensor_y).data
                loss_r_cos_pre_stage2 = self.criterion_cos(select_grasp_class_stage2[:,3:6], next_gt[class_select,3:6],tensor_y).data

                loss_theta_pre        = F.smooth_l1_loss(select_grasp_class[:,6], next_gt[class_select,6], reduction='mean').data
                loss_theta_pre_score  = F.smooth_l1_loss(select_grasp_score[:,6], next_gt[score_select,6], reduction='mean').data
                loss_theta_pre_stage2 = F.smooth_l1_loss(select_grasp_class_stage2[:,6], next_gt[class_select,6], reduction='mean').data

                loss_score_pre        = F.smooth_l1_loss(select_grasp_class[:,7:], next_gt[class_select,7:], reduction='mean').data
                loss_score_pre_score  = F.smooth_l1_loss(select_grasp_score[:,7:], next_gt[score_select,7:], reduction='mean').data
                loss_score_pre_stage2 = F.smooth_l1_loss(select_grasp_class_stage2[:,7:], next_gt[class_select,7:], reduction='mean').data

                print("loss stage 2 - class: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2))
                print("loss stage 3 - class: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre) )
                print("loss stage 3 - score: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score))

            TP = ((gt_class.view(-1) == 1 ) & (predit_formal.view(-1) == 1)).sum().float()
            TN = ((gt_class.view(-1) == 0 ) & (predit_formal.view(-1) == 0)).sum().float()
            FP = ((gt_class.view(-1) == 0 ) & (predit_formal.view(-1) == 1)).sum().float()
            FN = ((gt_class.view(-1) == 1 ) & (predit_formal.view(-1) == 0)).sum().float()
            print(TP,TN,FN,FP)
            acc = (TP + TN) / (TP + TN + FP + FN)
            correct_refine_tuple = (TP, TN, FP, FN)
            print("stage2 acc:", acc)

            loss_refine_tuple = (loss, loss_class.data, loss_grasp_center.data, loss_grasp_r.data, loss_grasp_theta.data, loss_grasp_score, \
                                                loss_center_pre_stage2, loss_r_cos_pre_stage2, loss_theta_pre_stage2, loss_score_pre_stage2,
                                                loss_center_pre, loss_r_cos_pre, loss_theta_pre, loss_score_pre,
                                                loss_center_pre_score, loss_r_cos_pre_score, loss_theta_pre_score, loss_score_pre_score)
                                    
        return select_grasp_class, select_grasp_score, select_grasp_class_stage2, class_select, score_select, loss_refine_tuple, correct_refine_tuple

    def refine_forward(self, pc_group_more_xyz, pc_group_more_index, true_mask, all_feature, \
                    group_feature_mp, next_grasp, gripper_params, next_gt=None):
        '''
          pc_group_more_xyz   :[B*center_num, group_num_more, 6]
          pc_group_more_index :[B, center_num, group_num_more]
          true_mask           :[len(true_mask)]
          all_feature         :[B, A, Feature]
          group_feature_mp    :[B*N_C, 128, 1]
          next_grasp          :[len(true_mask), 10]
          gripper_params      :List [torch.tensor(),float,float] widths, height, depth
          next_gt             :[len(true_mask), 10]
        '''
        B, feature_len = all_feature.shape[0], all_feature.shape[2]
        N_C, N_G_M = pc_group_more_index.shape[1], pc_group_more_index.shape[2]
        cuda = pc_group_more_xyz.is_cuda
        
        gripper_pc, gripper_pc_index, gripper_pc_index_inall, gripper_mask = get_gripper_region_transform(pc_group_more_xyz[true_mask], 
                        pc_group_more_index.view(-1,N_G_M)[true_mask], next_grasp, self.gripper_number, gripper_params)
        select_grasp_class, select_grasp_score, select_grasp_class_stage2 = None, None, None
        final_mask, final_mask_sthre, loss_refine_tuple, correct_refine_tuple = None, None, (None, None), (None, None)


        if len(gripper_mask) >= 2:
            all_feature_new = all_feature.contiguous().view(-1, feature_len)
            add = torch.arange(B).view(-1,1).repeat(1, N_C).view(-1)[true_mask].view(-1,1).repeat(1, self.gripper_number)
            if pc_group_more_index.cuda:
                add = add.cuda()
            #### gripper_pc_index_inall: [len(true_mask), region_num]
            gripper_pc_index_inall_new = (gripper_pc_index_inall.long() + add * all_feature.shape[1]).view(-1)
            gripper_feature = all_feature_new[gripper_pc_index_inall_new].view(-1, self.gripper_number, feature_len)[gripper_mask]#.detach()
            #### gripper_feature: [len(gripper_mask), self.gripper_number, feature_len]
            
            group_feature_mp = group_feature_mp.view(-1,128)[gripper_mask].contiguous()
            
            # next_x_cls: [len(gripper_mask), 2], next_x_reg: [len(gripper_mask), 10]
            next_x_cls, next_x_reg = self.extrat_feature_refine(gripper_feature.permute(0,2,1), group_feature_mp)
            if next_gt is not None:
                next_gt = next_gt[gripper_mask]
            select_grasp_class, select_grasp_score, select_grasp_class_stage2, class_select, score_select, loss_refine_tuple, \
                                correct_refine_tuple = self.compute_loss_refine(next_grasp[gripper_mask], next_x_cls, next_x_reg, next_gt)

            if next_gt is not None:
                next_gt = next_gt[class_select]

            final_mask = true_mask.clone()[gripper_mask][class_select] 
            final_mask_sthre = true_mask.clone()[gripper_mask][score_select] 

        return select_grasp_class, select_grasp_score, select_grasp_class_stage2, final_mask, \
                            final_mask_sthre, loss_refine_tuple, correct_refine_tuple, next_gt
        
    def forward(self, pc_group, pc_group_more, pc_group_index, pc_group_more_index, center_pc, \
                    center_pc_index, pc, all_feature, gripper_params, ground_grasp=None, data_path=None):
        '''
          pc_group            :[B, center_num, group_num, 6]
          pc_group_more       :[B, center_num, group_num_more, 6]
          pc_group_index      :[B, center_num, group_num]
          pc_group_more_index :[B, center_num, group_num_more]
          center_pc           :[B, center_num, 6]
          center_pc_index     :[B, center_num]
          pc                  :[B, A, 6]
          all_feature         :[B, A, Feature])
          gripper_params      :List [float,float,float] width, height, depth
          ground_grasp:       :[B,center_num,8] the labels of grasps (ground truth + score)
        '''
        B,N_C,N_G,C = pc_group.shape
        _,_,N_G_M,_ = pc_group_more.shape
        
        cuda = pc.is_cuda
        final_grasp, final_grasp_stage1 = torch.Tensor(), torch.Tensor()
        loss_tuple, loss_tuple_stage2 = (None, None), (None, None)

        anchors = self._enumerate_anchors(center_pc[:,:,:3].view(-1,3).float())  ## [B*center_num, 8, 7]
        anchor_number = anchors.shape[1]

        pc_group_xyz = pc_group[:,:,:,:6].clone().view(B*N_C,N_G,-1)
        pc_group_more_xyz = pc_group_more[:,:,:,:6].clone().view(B*N_C,-1,6)
        
        feature_len = all_feature.shape[2]
        all_feature_new = all_feature.contiguous().view(-1, feature_len)
        add = torch.arange(B).view(-1,1).repeat(1, N_C*N_G)
        if pc_group_index.is_cuda:
            add = add.cuda()
        pc_group_index_new = (pc_group_index.long().view(B, N_C*N_G) + add * all_feature.shape[1]).view(-1)
        center_feature = all_feature_new[pc_group_index_new].view(B, N_C, N_G, feature_len)
        center_feature = center_feature.view(-1, N_G, feature_len)#[true_mask]#.detach()
        
        ######--------------------------don't use grasp region--------------------------
        # all_feature_new = all_feature.view(-1, feature_len)
        # add = torch.arange(B).view(-1,1).repeat(1, N_C)
        # if pc_group_index.cuda:
        #     add = add.cuda()
        # center_pc_index_new = (center_pc_index.long() + add * all_feature.shape[1]).view(-1)
        # center_feature = all_feature_new[center_pc_index_new].view(B, N_C, feature_len)
        # center_feature = center_feature.view(-1, 1, feature_len)#[true_mask]#.detach()
        # # center_feature:[B*N_C, N_G, feature_len]      
        
        # x_cls:[B*N_C, num_anchor]  x_reg:[B*N_C, num_anchor, 8] 
        x_cls, x_reg, mp_center_feature = self.extrat_feature_region(center_feature.permute(0,2,1), None)
        
        # next_grasp: [len(true_mask), 8], next_gt: [len(true_mask), 8]
        next_grasp, loss_tuple, correct_tuple, next_gt, tt_pre, true_mask = self.compute_loss(x_reg, anchors, x_cls, ground_grasp)
        # print("true_mask",true_mask)
        keep_grasp_num_stage2 = [(torch.sum((true_mask<(i+1)*N_C) & (true_mask>=i*N_C))) for i in range(B)] 
        
        select_grasp_class, select_grasp_score, select_grasp_class_stage2, final_mask, final_mask_sthre, keep_grasp_num_stage3, \
            keep_grasp_num_stage3_score, loss_refine_tuple, correct_refine_tuple, gt = None, None, None, None, None, None, None, None, None, None

        if self.is_training_refine:
            select_grasp_class, select_grasp_score, select_grasp_class_stage2, final_mask, final_mask_sthre, loss_refine_tuple, \
                            correct_refine_tuple, gt = self.refine_forward(pc_group_more_xyz, pc_group_more_index, true_mask, \
                            all_feature, mp_center_feature, next_grasp.detach(), gripper_params, next_gt)

            if final_mask is not None:
                keep_grasp_num_stage3       = [(torch.sum((final_mask<(i+1)*N_C) & (final_mask>=i*N_C))) for i in range(B)] 
                keep_grasp_num_stage3_score = [(torch.sum((final_mask_sthre<(i+1)*N_C) & (final_mask_sthre>=i*N_C))) for i in range(B)] 
            else:
                keep_grasp_num_stage3       = [0 for i in range(B)] 
                keep_grasp_num_stage3_score = [0 for i in range(B)] 

        # print("!!!!!!!!!!!!!!!!!!!!",B, N_C)
        # print(keep_grasp_num_stage2)
        return next_grasp.detach(), keep_grasp_num_stage2, true_mask, loss_tuple, correct_tuple, next_gt, \
                select_grasp_class, select_grasp_score, select_grasp_class_stage2, keep_grasp_num_stage3, \
                keep_grasp_num_stage3_score, final_mask, final_mask_sthre, loss_refine_tuple, correct_refine_tuple, gt

def get_gripper_region_transform(group_points, group_index, grasp, region_num, gripper_params):
    '''
      Return the transformed local points in the closing area of gripper.
      Input: group_points: [B,G,6]
             group_index : [B,G]
             grasp:        [B,7]
             region_num:   the number of saved points in the closing area
      Return:    
             gripper_pc      :[B,region_num,group_points.shape[2]]
             gripper_pc_index:[B,region_num]
             true_mask_index :[B]
    '''
    widths, height, depths = gripper_params
    B, _ = grasp.shape
    center = grasp[:, 0:3].float()
    axis_y = grasp[:, 3:6].float()
    angle = grasp[:, 6].float()
    cuda = center.is_cuda

    cos_t, sin_t = torch.cos(angle), torch.sin(angle)
    # R1 = torch.zeros((B, 3, 3))
    # for i in range(B):
    #     r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
    #     R1[i,:,:] = r
    one, zero = torch.ones((B, 1), dtype=torch.float32), torch.zeros((B, 1), dtype=torch.float32)
    if cuda:
        one, zero = one.cuda(), zero.cuda()
    R1 = torch.cat( (cos_t.view(B,1), zero, -sin_t.view(B,1), zero, one, zero, sin_t.view(B,1), 
                        zero, cos_t.view(B,1)), dim=1).view(B,3,3)
    if cuda:
        R1=R1.cuda()

    norm_y = torch.norm(axis_y, dim=1).add_(1e-12)
    axis_y = torch.div(axis_y, norm_y.view(-1,1))
    if cuda:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).cuda()
    else:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float)
        
    axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
    norm_x = torch.norm(axis_x, dim=1).add_(1e-12)
    axis_x = torch.div(axis_x, norm_x.view(-1,1))
    if cuda:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

    axis_z = torch.cross(axis_x, axis_y, dim=1)
    norm_z = torch.norm(axis_z, dim=1)
    axis_z = torch.div(axis_z, norm_z.view(-1,1))
    if cuda:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).cuda()
    else:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float)
    matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
    if cuda:
        matrix = matrix.cuda()
    matrix = torch.bmm(matrix, R1)
    approach = matrix[:,:,0]
    norm_x = torch.norm(approach, dim=1).add_(1e-12)
    approach = torch.div(approach, norm_x.view(-1,1))
    if cuda:
        axis_y = axis_y.cuda()
        group_points = group_points.cuda()
        center = center.cuda()
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

    minor_normal = torch.cross(approach, axis_y, dim=1)
    matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1)), dim=2).permute(0,2,1)
    ## pcs_t: [B,G,3]
    pcs_t = torch.bmm(matrix, (group_points[:,:,:3].float() - \
                        center.view(-1,1,3).repeat(1, group_points.shape[1], 1).float()).permute(0,2,1)).permute(0,2,1)

    # torch.tensor [B,1]  or  float
    x_limit = depths.float().view(-1,1)/2 if type(depths) is torch.Tensor else depths / 2 
    z_limit = height/2 # float
    # torch.tensor [B,1]  or  float
    y_limit = widths.float().view(-1,1)/2 if type(widths) is torch.Tensor else widths / 2

    gripper_pc = torch.full((B,region_num,group_points.shape[2]),-1)
    #gripper_pc_formal = torch.full((B,region_num,group_points.shape[2]),-1)
    gripper_pc_index       = torch.full((B,region_num),-1)
    gripper_pc_index_inall = torch.full((B,region_num),-1)
    true_mask = torch.zeros((B))
    
    x1 = pcs_t[:,:,0] > 0
    x2 = pcs_t[:,:,0] < x_limit
    y1 = pcs_t[:,:,1] > -y_limit
    y2 = pcs_t[:,:,1] < y_limit
    z1 = pcs_t[:,:,2] > -z_limit
    z2 = pcs_t[:,:,2] < z_limit
    
    a = torch.cat((x1.view(B,-1,1), x2.view(B,-1,1), y1.view(B,-1,1), \
                    y2.view(B,-1,1), z1.view(B,-1,1), z2.view(B,-1,1)), dim=-1)
    for i in range(B):
        index = torch.nonzero(torch.sum(a[i], dim=-1) == 6).view(-1)
        if len(index) > region_num:
            #print(len(index))
            index = index[np.random.choice(len(index),region_num,replace=False)]
        elif len(index) > 5:
            index = index[np.random.choice(len(index),region_num,replace=True)]
        if len(index) > 5:
            gripper_pc[i] = torch.cat((pcs_t[i,index], group_points[i,index,3:]),-1)
            #gripper_pc_formal[i] = group_points[i,index,:]
            gripper_pc_index[i]       = index
            gripper_pc_index_inall[i] = group_index[i][index]
            true_mask[i] = 1
    
    if cuda:
        gripper_pc, true_mask, gripper_pc_index, gripper_pc_index_inall = gripper_pc.cuda(), true_mask.cuda(), \
                                                        gripper_pc_index.cuda(), gripper_pc_index_inall.cuda()#, gripper_pc_formal.cuda()
    true_mask_index = torch.nonzero(true_mask==1).view(-1)
    return gripper_pc, gripper_pc_index, gripper_pc_index_inall, true_mask_index#, gripper_pc_formal

def _enumerate_templates():
    '''
      Enumerate all grasp anchors:
      For one score center, we generate 120 anchors.

      grasp configuration:(p, r, theta)
      r -> (1,0,0),                   (sqrt(2)/2, 0, sqrt(2)/2),           (sqrt(2)/2, 0, -sqrt(2)/2),           \
           (sqrt(2)/2,sqrt(2)/2,0),   (sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),   (sqrt(3)/3, sqrt(3)/3, -sqrt(3)/3),   \
           (0,1,0),                   (0, sqrt(2)/2, sqrt(2)/2),           (0, sqrt(2)/2, -sqrt(2)/2),           \
           (-sqrt(2)/2,sqrt(2)/2,0),  (-sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),  (-sqrt(3)/3, sqrt(3)/3, -sqrt(3)/3),  \
           (-1,0,0),                  (-sqrt(2)/2, 0, sqrt(2)/2),          (-sqrt(2)/2, 0, -sqrt(2)/2),          \
           (-sqrt(2)/2,-sqrt(2)/2,0), (-sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3), (-sqrt(3)/3, -sqrt(3)/3, -sqrt(3)/3), \
           (0,-1,0),                  (0, -sqrt(2)/2, sqrt(2)/2),          (0, -sqrt(2)/2, -sqrt(2)/2),          \
           (sqrt(2)/2,-sqrt(2)/2,0),  (sqrt(3)/3, -sqrt(3)/3, sqrt(3)/3),  (sqrt(3)/3, -sqrt(3)/3, -sqrt(3)/3)
      theta -> {-pi/2, -pi/4, 0, pi/4, pi/2}
    '''
    sqrt2 = math.sqrt(2)/2
    sqrt3 = math.sqrt(3)/3
    t_r = torch.FloatTensor([
                        [sqrt3, sqrt3, sqrt3], [sqrt3, sqrt3, -sqrt3], \
                        [sqrt3, -sqrt3, -sqrt3], [sqrt3, -sqrt3, sqrt3]\
                        ]).view(1,4,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)
    #t_r = torch.FloatTensor([
    #                    [sqrt3, sqrt3, sqrt3], [sqrt3, sqrt3, -sqrt3], \
    #                    [-sqrt3, sqrt3, -sqrt3], [-sqrt3, sqrt3, sqrt3], \
    #                    [-sqrt3, -sqrt3, sqrt3], [-sqrt3,-sqrt3, -sqrt3], \
    #                    [sqrt3, -sqrt3, -sqrt3], [sqrt3,-sqrt3, sqrt3]\
    #                    ]).view(1,8,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)

    #t_r = torch.FloatTensor([#[1.0,0,0], [-1.0,0,0], [0,1.0,0], [0,-1.0,0],
    #                    [sqrt2, sqrt2, 0], [sqrt2, -sqrt2, 0]
    #                    ]).view(1,2,1,3).repeat(1,1,1,1)#repeat(1,1,5,1)
    #t_theta = torch.FloatTensor([-math.pi/4, 0, math.pi/4]).view(1,1,3,1).repeat(1,8,1,1)
    t_theta = torch.FloatTensor([0]).view(1,1,1,1).repeat(1,4,1,1)
    tem = torch.cat([t_r, t_theta], dim=3).half()
    return tem

def compute_cos_sim(a, b):
    '''
      input:
         a :[N, 3]
         b :[N, 3]
      output:
         sim :[N, 1]
    '''
    a_b = torch.sum(torch.mul(a, b), dim=1)
    epsilon = 1e-12
    a2 = torch.add(torch.sum(torch.mul(a, a), dim=1), (epsilon))
    b2 = torch.add(torch.sum(torch.mul(b, b), dim=1), (epsilon))
    div_ab = torch.sqrt(torch.mul(a2, b2))
    sim = torch.div(a_b, div_ab).mul_(-1).add_(1).view(-1,1)
    '''
    b_copy = -b
    a_b_copy = torch.sum(torch.mul(a, b_copy), dim=1)
    sim_copy = torch.div(a_b_copy, div_ab).mul_(-1).add_(1).view(-1,1)

    sim = torch.min(sim, sim_copy)
    '''
    return sim
    

if __name__ == '__main__':
    pass