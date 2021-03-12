import os
import numpy as np
import time
import pickle
import transforms3d

import torch
from tensorboardX import SummaryWriter

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from dataset_utils.scoredataset import ScoreDataset
from multi_model.score_network import ScoreNetwork
from multi_model.gripper_region_network import GripperRegionNetwork

from dataset_utils.eval_score.eval import eval_test, eval_validate

def mkdir_output(base_path, tag, mode="train", log_flag=False):
    path = os.path.join(base_path, tag)
    if not os.path.exists(path):
        os.mkdir(path)

    if log_flag:
        logger = SummaryWriter(path)
        return logger
      
def get_dataset(all_points_num, base_path, tag="train", seed=1, width=None):
    dataset = ScoreDataset(
                all_points_num = all_points_num,
                path = base_path,
                tag = tag,
                data_seed = seed,
                data_width = width)
    print(len(dataset))
    return dataset

def get_dataloader(dataset, batchsize, shuffle=True, num_workers=8, pin_memory=True):
    def my_worker_init_fn(pid):
        np.random.seed(torch.initial_seed() % (2**31-1))
    def my_collate(batch):
        batch = list(filter(lambda x:x[0] is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batchsize,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    shuffle=shuffle,
                    worker_init_fn=my_worker_init_fn,
                    collate_fn=my_collate,
                )
    return dataloader

def construct_scorenet(load_flag, obj_class_num=2, model_path=None, gpu_num=0):
    score_model = ScoreNetwork(training=True, k_obj=obj_class_num)

    resume_num = 0
    if load_flag and model_path is not '':
        model_dict = torch.load(model_path, map_location='cuda:{}'.format(gpu_num)).state_dict() #, map_location='cpu'
        new_model_dict = {}
        for key in model_dict.keys():
            new_model_dict[key.replace("module.", "")] = model_dict[key]
        score_model.load_state_dict(new_model_dict)
        resume_num = 1+int(model_path.split('/')[-1].split('_')[1].split('.model')[0])
    return score_model, resume_num

def construct_rnet(load_flag, training_refine, group_num, 
            gripper_num, grasp_score_threshold, depth, reg_channel, model_path=None, gpu_num=0):
    #-------------- load region (refine) network----------------
    region_model = GripperRegionNetwork(training=training_refine, group_num=group_num, \
        gripper_num=gripper_num, grasp_score_threshold=grasp_score_threshold, radius=depth, reg_channel=reg_channel)
    
    resume_num = 0

    if load_flag and model_path is not '':
        cur_dict = region_model.state_dict()                                        
        model_dict = torch.load(model_path, map_location='cuda:{}'.format(gpu_num)).state_dict()
        new_model_dict = {}
        for key in model_dict.keys():
            new_model_dict[key.replace("module.", "")] = model_dict[key]

        cur_dict.update(new_model_dict)
        region_model.load_state_dict(cur_dict)
        resume_num = 1+int(model_path.split('/')[-1].split('_')[1].split('.model')[0])
    return region_model, resume_num

def construct_net(params, mode, gpu_num=0, load_score_flag=True, 
                                score_path=None, load_rnet_flag=True, rnet_path=None):
    # mode = ['train', 'pretrain_score', 'pretrain_region', 'validate', \
    # 'validate_score', 'validate_region', 'test', 'test_score', 'test_region']
    obj_class_num, group_num, gripper_num, grasp_score_threshold, depth, reg_channel = params
    if 'validate' in mode or 'test' in mode or mode == 'train':
        load_score_flag, load_rnet_flag = True, True
    elif mode == 'pretrain_region':
        load_score_flag = True
        
    score_model, score_resume = construct_scorenet(load_score_flag, obj_class_num, 
                                                        score_path, gpu_num)
    if 'score' in mode:
        return score_model, None, score_resume
    elif 'region' in mode:
        training_refine = False
    else:
        training_refine = True

    region_model, rnet_resume = construct_rnet(load_rnet_flag, training_refine, group_num, 
                    gripper_num, grasp_score_threshold, depth, reg_channel, rnet_path, gpu_num)
    if mode == "train" and 'pretrain' in rnet_path.split('/')[-1]:
        rnet_resume = 0
    return score_model, region_model, rnet_resume

def construct_scheduler(model, lr, resume_num=0):
    optimizer = optim.Adam([{'params':model.parameters(), 'initial_lr':lr}], lr=lr)
    #print(resume_num)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=resume_num-1)
    return optimizer, scheduler

def map_model(score_model, region_model, gpu_num:int, gpu_id:int, gpu_ids:str):
    device = torch.device("cuda:"+str(gpu_id))
    score_model = score_model.to(device)
    if region_model is not None:
        region_model = region_model.to(device)
    
    if gpu_num > 1:
        device_id = [int(i) for i in gpu_ids.split(',')]
        score_model = nn.DataParallel(score_model, device_ids=device_id)
        if region_model is not None:
            region_model = nn.DataParallel(region_model, device_ids=device_id)
    print("Construct network successfully!")
    return score_model, region_model

def add_log_epoch(logger, data, epoch, mode="train", method="refine"):
    if method == "score":
        logger.add_scalar('epoch_'+mode+'_stage1_loss_score', data[0], epoch) # scorenet regression loss
    elif method == "region":
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_center', data[0], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_cos_orientation', data[1], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_theta', data[2], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_score', data[3], epoch)        
    elif method == "refine":
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_center', data[0], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_cos_orientation', data[1], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_theta', data[2], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage2_pre_loss_score', data[3], epoch)   

        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_center_stage2', data[4], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_cos_orientation_stage2', data[5], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_theta_stage2', data[6], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_score_stage2', data[7], epoch)    
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_center', data[8], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_cos_orientation', data[9], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_theta', data[10], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_score', data[11], epoch)    
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_center_score', data[12], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_cos_orientation_score', data[13], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_theta_score', data[14], epoch) 
        logger.add_scalar('epoch_'+mode+'_stage3_pre_loss_score_score', data[15], epoch)    

def add_log_batch(logger, data, index, mode="train", method="refine"):
    def add_log_stage1(logger, loss, index, mode):
        logger.add_scalar('batch_'+mode+'_stage1_loss_score', loss, index) # scorenet regression loss

    def add_log_stage2(logger, acc, loss_tuple, index, mode):
        logger.add_scalar('batch_'+mode+'_stage2_anchor_acc', acc, index)  # acc

        logger.add_scalar('batch_'+mode+'_stage2_loss', (loss_tuple[0].mean().data), index)
        logger.add_scalar('batch_'+mode+'_stage2_loss_class', (loss_tuple[1].mean()), index)                                           
        logger.add_scalar('batch_'+mode+'_stage2_loss_first1', (loss_tuple[2].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage2_loss_first2', (loss_tuple[3].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage2_loss_first3', (loss_tuple[4].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage2_loss_first4', (loss_tuple[5].mean()), index)
        
        logger.add_scalar('batch_'+mode+'_stage2_pre_loss_center', (loss_tuple[6].mean().data), index)
        logger.add_scalar('batch_'+mode+'_stage2_pre_loss_cos_orientation', (loss_tuple[7].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage2_pre_loss_theta', (loss_tuple[8].mean().data), index)  
        logger.add_scalar('batch_'+mode+'_stage2_pre_loss_score', (loss_tuple[9].mean().data), index)  
    
    def add_log_stage3(logger, acc_refine, loss_refine_tuple, index, mode):
        logger.add_scalar('batch_'+mode+'_stage3_refine_acc', acc_refine, index) # acc_refine

        logger.add_scalar('batch_'+mode+'_stage3_loss', (loss_refine_tuple[0].mean().data), index)
        logger.add_scalar('batch_'+mode+'_stage3_loss_class', (loss_refine_tuple[1].mean()), index)                                           
        logger.add_scalar('batch_'+mode+'_stage3_loss_first1', (loss_refine_tuple[2].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_loss_first2', (loss_refine_tuple[3].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_loss_first3', (loss_refine_tuple[4].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_loss_first4', (loss_refine_tuple[5].data.mean()), index)

        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_center_stage2', (loss_refine_tuple[6].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_cos_orientation_stage2', (loss_refine_tuple[7].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_theta_stage2', (loss_refine_tuple[8].mean().data), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_score_stage2', (loss_refine_tuple[9].mean().data), index) 

        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_center', (loss_refine_tuple[10].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_cos_orientation', (loss_refine_tuple[11].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_theta', (loss_refine_tuple[12].mean().data), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_score', (loss_refine_tuple[13].mean().data), index)

        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_center_score', (loss_refine_tuple[14].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_cos_orientation_score', (loss_refine_tuple[15].mean()), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_theta_score', (loss_refine_tuple[16].mean().data), index)
        logger.add_scalar('batch_'+mode+'_stage3_pre_loss_score_score', (loss_refine_tuple[17].mean().data), index)

    if method == "score":
        add_log_stage1(logger, data[0].mean().data, index, mode)
    
    elif method == "region":
        add_log_stage1(logger, data[0].mean().data, index, mode)

        loss_tuple = data[2]
        acc        = data[1].mean().data
        add_log_stage2(logger, acc, loss_tuple, index, mode)

    elif method == "refine":
        add_log_stage1(logger, data[0].mean().data, index, mode)

        loss_tuple, loss_refine_tuple = data[3], data[4]
        acc, acc_refine               = data[1].mean().data, data[2].mean().data
        add_log_stage2(logger, acc, loss_tuple, index, mode)
        add_log_stage3(logger, acc_refine, loss_refine_tuple, index, mode)

    if mode == "train":
        if method == "direct":
            logger.add_scalar('train_loss_first1', (loss_tuple[1].mean()), index)
            logger.add_scalar('train_loss_first2', (loss_tuple[2].mean()), index)
            logger.add_scalar('train_loss_first3', (loss_tuple[3].mean()), index)
            logger.add_scalar('train_loss_first4', (loss_tuple[4].mean()), index)
            logger.add_scalar('train_loss_cos_roi_pre', (loss_tuple[5].mean()), index)
            logger.add_scalar('train_loss_theta_pre', (loss_tuple[6].mean()), index)
            logger.add_scalar('train_loss_center_pre', (loss_tuple[7].mean().data), index)
            logger.add_scalar('train_loss_score_pre', (loss_tuple[8].mean().data), index)

def map_grasp_pc(data_path, keep_grasp_num, grasp, params, mask):
    '''
      Input:
        keep_grasp_num: the number of predicted grasps for each data_path
        data_path     : List
        grasp         : torch.Tensor [N, 8]
    '''
    print(keep_grasp_num)
    keep_grasp_num_new = keep_grasp_num[0].view(-1,1)
    #print(keep_grasp_num_new)
    for i in range(1, len(keep_grasp_num)):
        keep_grasp_num_new = torch.cat((keep_grasp_num_new, keep_grasp_num[i].view(-1,1)), dim=1)
    keep_grasp_num_new = keep_grasp_num_new.view(-1)
    for i in range(1, len(keep_grasp_num_new)): 
        keep_grasp_num_new[i] += keep_grasp_num_new[i-1]

    # print(len(data_path),len(keep_grasp_num_new))
    # print(keep_grasp_num_new)
    assert len(data_path) == len(keep_grasp_num_new)

    map_dict = {data_path[0]: grasp[:keep_grasp_num_new[0]]}
    for i in range(1, len(data_path)):
        map_dict.update({data_path[i]: grasp[keep_grasp_num_new[i-1]:keep_grasp_num_new[i]]})
    
    if type(params[0]) is float:
        return map_dict, None
    lengths = params[0].view(-1,1).repeat(1,params[-1]).view(-1)
    lengths = lengths[mask]
    map_param_dict = {data_path[0]: lengths[:keep_grasp_num_new[0]]}
    for i in range(1, len(data_path)):
        map_param_dict.update({data_path[i]: lengths[keep_grasp_num_new[i-1]:keep_grasp_num_new[i]]})
    return map_dict, map_param_dict
 
def eval_grasp_with_gt(map_dict, record_data, params, map_param_dict, score_thres=None):
    '''
        map_dict : {data_path: grasp}
    '''
    total_vgr, total_score, grasp_num, grasp_num_before = record_data
    batch_vgr, batch_score, batch_vgr_before, batch_grasp_num, batch_grasp_before_num = 0, 0, 0, 0, 0
    total_vgrs, total_scores, total_grasp_nums, total_grasp_before_nums = None, None, None, None
    depths, width, table_height, gpu, N_C = params
    
    for cur_data_path in map_dict.keys():
        cur_grasp = map_dict[cur_data_path]
        # print("formal grasp number: {}".format(len(cur_grasp)))
        if len(cur_grasp) <= 0:
            continue
        cur_data = np.load(cur_data_path, allow_pickle=True)
        if '0' in cur_data_path.split('/')[-3]:
            width = float(cur_data_path.split('/')[-3])  # eg. .../4080_view_1.p

        if 'noise' not in cur_data_path.split('_')[-1]:
            view_num = int(cur_data_path.split('_')[-1].split('.')[0])  # eg. .../4080_view_1.p
        else:
            view_num = int(cur_data_path.split('_')[-2])              # eg. .../4080_view_1_noise.p
        
        depths = depths if map_param_dict is None else map_param_dict[cur_data_path]
        vgr, score, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = \
                eval_validate(cur_data, cur_grasp[:,:8], view_num, table_height, depths, width, gpu)

        total_vgr           += vgr
        total_score         += score
        grasp_num           += grasp_nocoll_view_num
        grasp_num_before    += len(cur_grasp)

        batch_vgr              += vgr
        batch_score            += score
        batch_grasp_num        += grasp_nocoll_view_num
        batch_grasp_before_num += len(cur_grasp)

        if grasp_nocoll_view_num == 0:
            grasp_nocoll_view_num = 1
        print("before vgr: {}".format(vgr/len(cur_grasp)) )
        print("vgr: {}".format(vgr/grasp_nocoll_view_num) )
        print("score: {}".format(score/grasp_nocoll_view_num) )

    if batch_grasp_before_num == 0:
        batch_grasp_before_num = 1
    if batch_grasp_num == 0:
        batch_grasp_num = 1
    batch_vgr_before = batch_vgr / batch_grasp_before_num
    batch_vgr           /= batch_grasp_num
    batch_score         /= batch_grasp_num
    print("#before batch vgr \t", batch_vgr_before)
    print("#batch vgr \t", batch_vgr)
    print("#batch score \t", batch_score)
    record_data = (total_vgr, total_score, grasp_num, grasp_num_before)

    if score_thres:
        total_vgrs, total_scores, total_grasp_nums, total_grasp_before_nums = \
            [0]*len(score_thres), [0]*len(score_thres), [0]*len(score_thres), [0]*len(score_thres), [0]*len(score_thres)
        for ind in range(len(score_thres)):
            score_thre = score_thres[ind]
            for cur_data_path in map_dict.keys():
                cur_grasp = map_dict[cur_data_path]
                cur_grasp = cur_grasp(cur_grasp[:,7] > score_thre)
                # print("formal grasp number: {}".format(len(cur_grasp)))
                if len(cur_grasp) <= 0:
                    continue
                cur_data = np.load(cur_data_path, allow_pickle=True)
                if '0' in cur_data_path.split('/')[-3]:
                    width = float(cur_data_path.split('/')[-3])  # eg. .../4080_view_1.p

                if 'noise' not in cur_data_path.split('_')[-1]:
                    view_num = int(cur_data_path.split('_')[-1].split('.')[0])  # eg. .../4080_view_1.p
                else:
                    view_num = int(cur_data_path.split('_')[-2])              # eg. .../4080_view_1_noise.p
                
                depths = depths if map_param_dict is None else map_param_dict[cur_data_path]
                vgr, score, grasp_nocoll_view_num, grasp_nocoll_view, grasp_nocoll_scene = \
                        eval_validate(cur_data, cur_grasp[:,:8], view_num, table_height, depths, width, gpu)

                total_vgrs[ind]              += vgr
                total_scores[ind]            += score
                total_grasp_nums[ind]        += grasp_nocoll_view_num
                total_grasp_before_nums[ind] += len(cur_grasp)

    return batch_vgr, batch_score, batch_vgr_before, record_data, \
            total_vgrs, total_scores, total_grasp_nums, total_grasp_before_nums

def eval_and_log(logger, data_path, keep_grasp_num, mask, grasp, record_data, params, index, mode, stage='stage2'):
    map_dict, map_param_dict = map_grasp_pc(data_path, keep_grasp_num, grasp, params, mask)
    print("=======================evaluate grasp from {}=======================".format(stage))
    score_thres = None
    # if 'test' in mode and stage == 'stage3_class':
    #     score_thres = [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
    batch_vgr, batch_score, batch_vgr_before, record_data, \
            total_vgrs, total_scores, total_grasp_nums, total_grasp_before_nums = \
            eval_grasp_with_gt(map_dict, record_data, params, map_param_dict, score_thres)

    if batch_vgr !=0 and batch_score !=0 and batch_vgr_before !=0 :
        logger.add_scalar('batch_'+mode+'_vgr_'+stage, batch_vgr, index)
        logger.add_scalar('batch_'+mode+'_score_'+stage, batch_score, index)
        logger.add_scalar('batch_'+mode+'_vgr_before_'+stage, batch_vgr_before, index)
    print("=========================================================================")
    return record_data

def add_eval_log_epoch(logger, data, batch_nums, epoch, mode, stages):
    for i in range(len(data)):
        data_i  = data[i]
        stage_i = stages[i]
        #### data_i :(nocoll_scene_num, total_score, nocoll_view_num, formal_num)
        total_vgr_before    = data_i[0] / data_i[3]
        total_score         = data_i[1] / data_i[2]
        total_vgr           = data_i[0] / data_i[2]
        print("{} before_total_vgr: \t{}".format(stage_i, total_vgr_before) )
        print("{} total_vgr: \t{}".format(stage_i, total_vgr) )
        print("{} total_score: \t{}".format(stage_i, total_score) )

        logger.add_scalar('epoch_'+mode+'_'+stage_i+'_vgr_before', total_vgr_before, epoch) 
        logger.add_scalar('epoch_'+mode+'_'+stage_i+'_vgr', total_vgr, epoch) 
        logger.add_scalar('epoch_'+mode+'_'+stage_i+'_score', total_score, epoch) 

####___________________________test function_____________________________
def eval_notruth(pc, color, grasp_stage2, grasp_stage3, grasp_stage3_score, grasp_stage3_stage2, output_score, params, grasp_save_path=None):
    depths, width, table_height, gpu, _ = params
    view_num = None
    if len(grasp_stage2) >= 1:
        grasp_stage2 = eval_test(pc, grasp_stage2[:,:8], view_num, table_height, depths, width, gpu)
    if len(grasp_stage3_stage2) >= 1:
        grasp_stage3_stage2 = eval_test(pc, grasp_stage3_stage2[:,:8], view_num, table_height, depths, width, gpu)
    if len(grasp_stage3) >= 1:
        grasp_stage3 = eval_test(pc, grasp_stage3[:,:8], view_num, table_height, depths, width, gpu)
    if len(grasp_stage3_score) >= 1:
        grasp_stage3_score = eval_test(pc, grasp_stage3_score[:,:8], view_num, table_height, depths, width, gpu)

    if gpu != -1:
        output_score        = output_score.view(-1,1).cpu()
        grasp_stage2        = grasp_stage2.cpu()
        grasp_stage3_stage2 = grasp_stage3_stage2.cpu()
        grasp_stage3        = grasp_stage3.cpu()
        grasp_stage3_score  = grasp_stage3_score.cpu()
    print("stage2 grasp num:", len(grasp_stage2))
    print("stage3 grasp num:", len(grasp_stage2))
    print("stage3 grasp num (with scorethre):", len(grasp_stage3_score))
    output_dict = {
        'points'             : pc,
        'colors'             : color,
        'scores'             : output_score.numpy(),
        'grasp_stage2'       : grasp_stage2.numpy(),
        'grasp_stage3_stage2': grasp_stage3_stage2.numpy(),
        'grasp_stage3'       : grasp_stage3.numpy(),
        'grasp_stage3_score' : grasp_stage3_score.numpy(),
    }
    print(grasp_save_path)
    if grasp_save_path:
        with open(grasp_save_path, 'wb') as file:
            pickle.dump(output_dict, file)

def noise_color(color):
    obj_color_time = 1-np.random.rand(3) / 5
    print("noise color time", obj_color_time)
    for i in range(3):
        color[:,i] *= obj_color_time[i]
    return color

def local_to_global_transformation_quat(point):
    T_local_to_global = np.eye(4)
    #quat = transforms3d.quaternions.axangle2quat([1,0,0], np.pi*1.13)
    quat = transforms3d.euler.euler2quat(-0.87*np.pi, 0, 0)
    frame = transforms3d.quaternions.quat2mat(quat)
    T_local_to_global[0:3, 0:3] = frame
    T_local_to_global[0:3, 3] = point
    return T_local_to_global  

def transform_grasp(grasp_ori):
    '''
      Input:
        grasp_ori: [B, 13] 
      Output:
        grasp_trans:[B, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
    '''
    B, _ = grasp_ori.shape
    grasp_trans = torch.full((B, 8), -1)

    # axis_x = torch.cat([grasp_ori[:,0:1], grasp_ori[:,4:5], grasp_ori[:,8:9]], dim=1)
    # axis_y = torch.cat([grasp_ori[:,1:2], grasp_ori[:,5:6], grasp_ori[:,9:10]], dim=1)
    # axis_z = torch.cat([grasp_ori[:,2:3], grasp_ori[:,6:7], grasp_ori[:,10:11]], dim=1)
    # center = torch.cat([grasp_ori[:,3:4], grasp_ori[:,7:8], grasp_ori[:,11:12]], dim=1)
    axis_x, axis_y, axis_z, center = grasp_ori[:,0:3], grasp_ori[:,3:6], grasp_ori[:,6:9], grasp_ori[:,9:12]
    grasp_angle = torch.atan2(axis_x[:,2], axis_z[:,2])  ## torch.atan(torch.div(axis_x[:,2], axis_z[:,2])) is not OK!!!

    '''
    grasp_angle[axis_y[:,0] < 0] = np.pi-grasp_angle[axis_y[:,0] < 0]
    axis_y[axis_y[:,0] < 0] = -axis_y[axis_y[:,0] < 0]

    grasp_angle[grasp_angle >= 2*np.pi] = grasp_angle[grasp_angle >= 2*np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -2*np.pi] = grasp_angle[grasp_angle <= -2*np.pi] + 2*np.pi
    grasp_angle[grasp_angle > np.pi] = grasp_angle[grasp_angle > np.pi] - 2*np.pi
    grasp_angle[grasp_angle <= -np.pi] = grasp_angle[grasp_angle <= -np.pi] + 2*np.pi
    '''

    grasp_trans[:,:3]  = center
    grasp_trans[:,3:6] = axis_y
    grasp_trans[:,6]   = grasp_angle
    grasp_trans[:,7]   = grasp_ori[:,12]
    return grasp_trans
