import argparse
import os
import pickle, copy

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import open3d
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import time

from dataset_utils.get_regiondataset import get_grasp_allobj
from dataset_utils.eval_score.eval import eval_test, eval_validate
import utils
import glob

parser = argparse.ArgumentParser(description='GripperRegionNetwork')
parser.add_argument('--tag', type=str, default='default')
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--epoch', type=int, default=101)
                                        
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu-num', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=str, default='0,2,3')
parser.add_argument('--lr-score' , type=float, default=0.001)
parser.add_argument('--lr-region', type=float, default=0.001)

parser.add_argument('--load-score-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/models/pretrain_for_refine_regnet_bs15/score_17.model')
parser.add_argument('--load-region-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/models/pretrain_for_refine_regnet_bs15/region_17.model')
#parser.add_argument('--load-score-path', type=str, default='')
# parser.add_argument('--load-region-path', type=str, default='')
parser.add_argument('--load-score-flag', type=bool, default=True)
parser.add_argument('--load-region-flag', type=bool, default=True)

parser.add_argument('--data-path', type=str, default='/data1/cxg6/eval_data', help='data path')

parser.add_argument('--model-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/models/', help='to saved model path')
parser.add_argument('--log-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/log/', help='to saved log path')
parser.add_argument('--folder-name', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/test_file/virtual_data')
parser.add_argument('--file-name', type=str, default='00001_view_1.p')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--save-interval', type=int, default=1)


args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

np.random.seed(int(time.time()))
if args.cuda:
    torch.cuda.manual_seed(1)
torch.cuda.set_device(args.gpu)

all_points_num = 25600
obj_class_num = 43

# width, height, depth = 0.060, 0.010, 0.060
width, height, depth = 0.060, 0.010, 0.065
table_height = 0.5
grasp_score_threshold = 0.5 # 0.3
center_num = 1024#64#128
score_thre = 0.5
group_num=256
group_num_more=1024
r_time_group = 0.1
r_time_group_more = 0.8
gripper_num = 64
use_theta = True
reg_channel = 8 
        
gripper_params = [width, height, depth]
model_params   = [obj_class_num, group_num, gripper_num, grasp_score_threshold, depth, reg_channel]
params         = [center_num, score_thre, group_num, r_time_group, \
                    group_num_more, r_time_group_more, width, height, depth]

score_model, region_model, resume_epoch = utils.construct_net(model_params, 'test_one_file', gpu_num=args.gpu, 
                                load_score_flag=args.load_score_flag, score_path=args.load_score_path,
                                load_rnet_flag=args.load_region_flag, rnet_path=args.load_region_path)
score_model, region_model = utils.map_model(score_model, region_model, args.gpu_num, args.gpu, args.gpus)

class RefineModule():
    def __init__(self):
        self.eval_params    = [depth, width, table_height, args.gpu, center_num]
        self.params         = params
        self.gripper_params = gripper_params

    def test_one_file(self, epoch, pc_path, real_data=True):
        print("---------------ONE FILE: test_score epoch", epoch, "------------------")
        # np.random.seed(1)
        score_model.eval()
        region_model.eval()
        torch.set_grad_enabled(False)

        if real_data:
            data = open3d.io.read_point_cloud(pc_path)
            data.transform(utils.local_to_global_transformation_quat(center_camera))
            pc = np.array(data.points)
            pc_color = np.array(data.colors)
            center_camera = np.array([0, 0, 1.658])
        else:
            data = np.load(pc_path, allow_pickle=True)
            pc = data['view_cloud'].astype(np.float32)
            pc_color = data['view_cloud_color'].astype(np.float32)
        pc_back, color_back = copy.deepcopy(pc), copy.deepcopy(pc_color)
        # if real_data:
        #     center_camera = np.array([0, 0, 1.658])
        #     pc = np.matmul(utils.local_to_global_transformation_quat(center_camera), np.c_[pc, np.ones([len(pc), 1])].T).T[:,:3]
            
        pc_color = utils.noise_color(pc_color)
        pc = np.c_[pc, pc_color]
        if real_data:        
            pc = pc[pc[:,0] < 0.23]
            pc = pc[pc[:,0] > -0.5]
            pc = pc[pc[:,2] < 1]
            pc = pc[pc[:,1] < 0.7]
            pc = pc[pc[:,1] > 0.2]
        select_point_index = None
        if len(pc) >= all_points_num:
            select_point_index = np.random.choice(len(pc), all_points_num, replace=False)
        elif len(pc) < all_points_num:
            select_point_index = np.random.choice(len(pc), all_points_num, replace=True)
        pc = pc[select_point_index]

        pc_torch = torch.Tensor(pc).view(1, -1, 6)
        if args.gpu != -1:
            pc_torch = pc_torch.cuda()
        
        # all_feature: [B, N, C], output_score: [B, N]
        all_feature, output_score, _ = score_model(pc_torch)
        center_pc, center_pc_index, pc_group_index, pc_group, pc_group_more_index, \
                pc_group_more, _ = get_grasp_allobj(pc_torch, output_score, self.params, [], use_theta)

        grasp_stage2, keep_grasp_num_stage2, stage2_mask, _, _, _, select_grasp_class, select_grasp_score, \
                    select_grasp_class_stage2, keep_grasp_num_stage3, keep_grasp_num_stage3_score, \
                    stage3_mask, stage3_score_mask, _, _, _ = region_model(pc_group, pc_group_more, pc_group_index, \
                    pc_group_more_index, center_pc, center_pc_index, pc_torch, all_feature, self.gripper_params, None, [])
        
        grasp_save_path = pc_path.replace('_data', '_data_predict')
        if real_data:
            grasp_save_path = grasp_save_path.replace('.pcd', '.p')
        
        record_stage2 = utils.eval_notruth(pc_back, color_back, grasp_stage2, select_grasp_class, select_grasp_score, \
                                        select_grasp_class_stage2, output_score, self.eval_params, grasp_save_path)

def main():
    refineModule = RefineModule()
    real_data = True if 'real_data' in args.folder_name else False

    if not args.file_name:
        if real_data:
            pc_paths = glob.glob(args.folder_name+"/*.pcd",recursive=True)
        else:
            pc_paths = glob.glob(args.folder_name+"/*.p",recursive=True)

        for pc_path in pc_paths:
            refineModule.test_one_file(resume_epoch, pc_path, real_data)
    else:
        pc_path = os.path.join(args.folder_name, args.file_name)
        refineModule.test_one_file(resume_epoch, pc_path, real_data)

if __name__ == "__main__":
    main()
