import argparse
import os
import pickle

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
parser.add_argument('--mode', choices=['train', 'pretrain_score', 'pretrain_region', 'validate', 'validate_score',
                                        'validate_region', 'test', 'test_score', 'test_region'], required=True)
parser.add_argument('--batch-size', type=int, default=12)#16)#16
parser.add_argument('--num-refine', type=int, default=0, help='number of interative refinement iterations')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu-num', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=str, default='0,2,3')
parser.add_argument('--lr-score' , type=float, default=0.001) #0.001
parser.add_argument('--lr-region', type=float, default=0.001)

#parser.add_argument('--load-score-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/models/region_xyz/score_16.model')
# parser.add_argument('--load-score-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/models/train_direct/score_7.model')
# parser.add_argument('--load-region-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/models/train_direct/region_7.model')
parser.add_argument('--load-score-path', type=str, default='')
parser.add_argument('--load-region-path', type=str, default='')
parser.add_argument('--load-score-flag', type=bool, default=True)
parser.add_argument('--load-region-flag', type=bool, default=True)

parser.add_argument('--use-multi', type=bool, default=False)
parser.add_argument('--data-path', type=str, default='/data1/cxg6/dataset/0.08', help='data path')

parser.add_argument('--model-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/models/', help='to saved model path')
parser.add_argument('--log-path', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/assets/log/', help='to saved log path')
parser.add_argument('--folder-name', type=str, default='/data1/cxg6/REGNet_for_3D_Grasping/test_file/real_data')
parser.add_argument('--file-name', type=str, default='')
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--save-interval', type=int, default=1)


args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

np.random.seed(int(time.time()))
if args.cuda:
    torch.cuda.manual_seed(1)
torch.cuda.set_device(args.gpu)

if args.mode == 'test' or args.mode == 'validate':
    args.tag = args.load_score_path.split('/')[-2]

logger = utils.mkdir_output(args.log_path, args.tag, args.mode, log_flag=True)
utils.mkdir_output(args.model_path, args.tag)

all_points_num = 25600
obj_class_num = 43

# width, height, depth = 0.060, 0.010, 0.060
width, height, depth = 0.08, 0.010, 0.06#5
table_height = 0.75
grasp_score_threshold = 0.5 # 0.3
center_num = 64#64#128
score_thre = 0.5
group_num=256
group_num_more=1024
r_time_group = 0.1
r_time_group_more = 0.8
gripper_num = 64
use_theta = True
reg_channel = 10 
        
gripper_params = [width, height, depth]
model_params   = [obj_class_num, group_num, gripper_num, grasp_score_threshold, depth, reg_channel]
params         = [center_num, score_thre, group_num, r_time_group, \
                    group_num_more, r_time_group_more, width, height, depth]
 
train_score_dataset = utils.get_dataset(all_points_num, args.data_path, "train", 1, width)   #Train
val_score_dataset = utils.get_dataset(all_points_num, args.data_path, "validate", 1, width)  #Validation
test_score_dataset = utils.get_dataset(all_points_num, args.data_path, "test", 1, width)     #Test

train_score_loader = utils.get_dataloader(train_score_dataset, args.batch_size, shuffle=True)
val_score_loader   = utils.get_dataloader(val_score_dataset, 1, shuffle=True)
test_score_loader  = utils.get_dataloader(test_score_dataset, 1, shuffle=False)

score_model, region_model, resume_epoch = utils.construct_net(model_params, args.mode, gpu_num=args.gpu, 
                                load_score_flag=args.load_score_flag, score_path=args.load_score_path,
                                load_rnet_flag=args.load_region_flag, rnet_path=args.load_region_path)
score_model, region_model = utils.map_model(score_model, region_model, args.gpu_num, args.gpu, args.gpus)

optimizer_score, scheduler_score = utils.construct_scheduler(score_model, args.lr_score, resume_epoch)
if region_model is not None:
    optimizer_region, scheduler_region = utils.construct_scheduler(region_model, args.lr_region, resume_epoch)
    print(optimizer_region)

class ScoreModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, val_data_loader=None, test_data_loader=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        
        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def train_val(self, epoch, mode='train'):
        if mode == 'train':
            score_model.train()
            #scheduler_score.step()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader
            batch_size = args.batch_size

        else:
            score_model.eval()
            torch.set_grad_enabled(False)
            if mode == 'validate':
                dataloader = self.val_data_loader
            elif mode == 'test':
                dataloader = self.test_data_loader
            batch_size = 1

        total_all_loss = 0
        for batch_idx, (pc, pc_score, pc_label, data_path, data_width) in enumerate(dataloader):
            if mode == 'train':
                optimizer_score.zero_grad()

            if args.gpu != -1:
                pc, pc_score, pc_label, data_width = pc.cuda(), pc_score.cuda(), pc_label.cuda(), data_width.cuda()
            all_feature, output_score, loss = score_model(pc, pc_score, pc_label)

            loss_total = loss.sum() 
            total_all_loss += loss.mean()
            if mode == 'train':
                loss_total.backward()
                optimizer_score.step()

            data = (loss,)
            utils.add_log_batch(logger, data, batch_idx + epoch * len(dataloader), mode=mode, method="score")
            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(mode, epoch, 
                        batch_idx * batch_size, len(dataloader.dataset), 100. * batch_idx * 
                        batch_size / len(dataloader.dataset), loss_total.data, args.tag))
        data = (total_all_loss.data/batch_idx,)
        utils.add_log_epoch(logger, data, epoch, mode=mode, method="score")
        
        if mode == 'train':
            scheduler_score.step()

    def pretrain_score(self, epoch):
        self.train_val(epoch, mode='train')

    def validate_score(self, epoch):
        self.train_val(epoch, mode='validate')

    def test_score(self, epoch):
        self.train_val(epoch, mode='test')
    
    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print("---------------pretrain_score epoch", epoch, "------------------")
            path_score = os.path.join(self.saved_base_path, 'score_{}.model'.format(epoch))

            self.pretrain_score(epoch)
            torch.save(score_model, path_score)

            self.validate_score(epoch)
            #self.test_score(epoch)
    
    def validate(self, epoch):
        print("---------------validate_score epoch", epoch, "------------------")
        self.validate_score(epoch)

    def test(self, epoch):
        print("---------------DATALOADER: test_score epoch", epoch, "------------------")
        self.test_score(epoch)

class RegionModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, val_data_loader=None, test_data_loader=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, width, table_height, args.gpu, center_num]
        self.params         = params
        self.gripper_params = gripper_params
        
        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def train_val(self, epoch, mode='train', use_log=True):
        if mode == 'train':
            score_model.train()
            region_model.train()
            # scheduler_score.step()
            # scheduler_region.step()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader   
            batch_size = args.batch_size

        else:
            score_model.eval()
            region_model.eval()
            torch.set_grad_enabled(False)
            if mode == 'validate':
                dataloader = self.val_data_loader
            elif mode == 'test':
                dataloader = self.test_data_loader
            batch_size = 1

        pre_loss1, pre_loss2, pre_loss3, pre_loss4 = 0, 0, 0, 0
        record_stage2 = (0, 0, 0, 0)

        for batch_idx, (pc, pc_score, pc_label, data_path, data_width) in enumerate(dataloader):
            if mode == 'train':
                optimizer_score.zero_grad()
                optimizer_region.zero_grad()

            if args.gpu != -1:
                pc, pc_score, pc_label, data_width = pc.cuda(), pc_score.cuda(), pc_label.cuda(), data_width.cuda()
    
            # all_feature: [B, N, C(512)], output_score: [B, N]
            all_feature, output_score, loss = score_model(pc, pc_score, pc_label)
            center_pc, center_pc_index, pc_group_index, pc_group, pc_group_more_index, pc_group_more, \
                    grasp_labels = get_grasp_allobj(pc, output_score, self.params, data_path, use_theta)
                
            grasp_stage2, keep_grasp_num_stage2, stage2_mask, loss_tuple, correct_tuple, next_gt, _, _, _, _, _, _, _, _, _, _, = \
                            region_model(pc_group, pc_group_more, pc_group_index, pc_group_more_index, center_pc,\
                            center_pc_index, pc, all_feature, self.gripper_params, grasp_labels, data_path)
            
            loss_total = (loss.sum() + loss_tuple[0].sum())  
            pre_loss1 += loss_tuple[6].mean() 
            pre_loss2 += loss_tuple[7].mean() 
            pre_loss3 += loss_tuple[8].mean() 
            pre_loss4 += loss_tuple[9].mean() 

            if mode == 'train':
                loss_total.backward()
                optimizer_score.step()
                optimizer_region.step()

            acc = correct_tuple[0].sum()  / (correct_tuple[0].sum()  + correct_tuple[1].sum() ) 

            data = (loss, acc, loss_tuple)
            utils.add_log_batch(logger, data, batch_idx + epoch * len(dataloader), mode=mode, method="region")

            if use_log:
                record_stage2 = utils.eval_and_log(logger, data_path, keep_grasp_num_stage2, stage2_mask, grasp_stage2, \
                                record_stage2, self.eval_params, batch_idx + epoch*len(dataloader), mode, stage='stage2')

            print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(mode,
                        epoch, batch_idx * batch_size, len(dataloader.dataset),
                        100. * batch_idx * batch_size / len(dataloader.dataset), loss_total.data, args.tag))
        
        data = (pre_loss1.data/batch_idx, pre_loss2.data/batch_idx,
                    pre_loss3.data/batch_idx, pre_loss4.data/batch_idx)
        utils.add_log_epoch(logger, data, epoch, mode=mode, method="region")
        if use_log:
            records = [record_stage2]
            stages  = ['stage2']
            utils.add_eval_log_epoch(logger, records, len(dataloader.dataset), epoch, mode, stages)
        if mode == 'train':
            scheduler_score.step()
            scheduler_region.step()

    def pretrain_region(self, epoch):
        self.train_val(epoch, mode='train', use_log=False)

    def validate_region(self, epoch):
        self.train_val(epoch, mode='validate', use_log=True)

    def test_region(self, epoch):
        self.train_val(epoch, mode='test', use_log=True)

    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print("---------------pretrain_region epoch", epoch, "------------------")
            path_score  = os.path.join(self.saved_base_path, 'score_{}.model'.format(epoch))
            path_region = os.path.join(self.saved_base_path, 'region_{}.model'.format(epoch))
            
            self.pretrain_region(epoch)
            torch.save(score_model, path_score)
            torch.save(region_model, path_region)
            self.validate_region(epoch)
            #self.test_region(epoch)

    def validate(self, epoch):
        print("---------------validate_score epoch", epoch, "------------------")
        self.validate_region(epoch)

    def test(self):
        print("---------------DATALOADER: test_score epoch", epoch, "------------------")
        self.test_region(epoch)

class RefineModule():
    def __init__(self, start_epoch=0, end_epoch=1, train_data_loader=None, val_data_loader=None, test_data_loader=None):
        self.start_epoch = start_epoch
        self.end_epoch   = end_epoch
        self.train_data_loader = train_data_loader 
        self.val_data_loader   = val_data_loader  
        self.test_data_loader  = test_data_loader
        self.eval_params    = [depth, width, table_height, args.gpu, center_num]
        self.params         = params
        self.gripper_params = gripper_params
        
        self.saved_base_path = os.path.join(args.model_path, args.tag)

    def train_val(self, epoch, mode='train', use_log=True):
        if mode == 'train':
            score_model.train()
            region_model.train()
            torch.set_grad_enabled(True)
            dataloader = self.train_data_loader
            batch_size = args.batch_size

        else:
            score_model.eval()
            region_model.eval()
            torch.set_grad_enabled(False)
            if mode == 'validate':
                dataloader = self.val_data_loader
            else:
                dataloader = self.test_data_loader
            batch_size = 1

        pre_loss1_stage2, pre_loss2_stage2, pre_loss3_stage2, pre_loss4_stage2 = 0, 0, 0, 0
        pre_loss1_stage3_class, pre_loss2_stage3_class, pre_loss3_stage3_class, pre_loss4_stage3_class = 0, 0, 0, 0
        pre_loss1_stage3_class_satge2, pre_loss2_stage3_class_satge2, pre_loss3_stage3_class_satge2, \
                                                            pre_loss4_stage3_class_satge2 = 0, 0, 0, 0
        pre_loss1_stage3_score, pre_loss2_stage3_score, pre_loss3_stage3_score, pre_loss4_stage3_score = 0, 0, 0, 0

        record_stage2, record_stage3, record_stage3_stage2, record_stage3_score = (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)
        
        for batch_idx, (pc, pc_score, pc_label, data_path, data_width) in enumerate(dataloader):
            if mode == 'train':
                optimizer_score.zero_grad()
                optimizer_region.zero_grad()

            t1 = time.time()
            if args.gpu != -1:
                # pc, pc_score, pc_label = pc, pc_score, pc_label
                pc, pc_score, pc_label, data_width = pc.cuda(), pc_score.cuda(), pc_label.cuda(), data_width.cuda()

            # all_feature: [B, N, C(512)], output_score: [B, N]
            all_feature, output_score, loss = score_model(pc, pc_score, pc_label)

            center_pc, center_pc_index, pc_group_index, pc_group, pc_group_more_index, pc_group_more, \
                    grasp_labels = get_grasp_allobj(pc, output_score, self.params, data_path, use_theta)
            
            try:
                grasp_stage2, keep_grasp_num_stage2, stage2_mask, loss_tuple, correct_tuple, next_gt, select_grasp_class, select_grasp_score, \
                    select_grasp_class_stage2, keep_grasp_num_stage3, keep_grasp_num_stage3_score, stage3_mask, stage3_score_mask, \
                    loss_refine_tuple, correct_refine_tuple, gt = region_model(pc_group, pc_group_more, pc_group_index, \
                    pc_group_more_index, center_pc, center_pc_index, pc, all_feature, self.gripper_params, grasp_labels, data_path)

                loss_total = loss.sum() + loss_tuple[0].sum()
                if len(loss_refine_tuple) > 2:
                    loss_total += loss_refine_tuple[0].sum()
                t2 = time.time()  
                print("forward time:", t2-t1) 
                if mode == 'train':
                    t1 = time.time()
                    loss_total.backward()
                    # if len(loss_refine_tuple) > 2:
                    #     loss_total = loss_refine_tuple[0].sum()
                    #     loss_total.backward()

                    t2 = time.time()
                    print("backward time:", t2-t1)
                    optimizer_score.step()
                    optimizer_region.step()

                pre_loss1_stage2 += loss_tuple[6].mean().data
                pre_loss2_stage2 += loss_tuple[7].mean().data 
                pre_loss3_stage2 += loss_tuple[8].mean().data 
                pre_loss4_stage2 += loss_tuple[9].mean().data 

                pre_loss1_stage3_class += loss_refine_tuple[10].mean().data
                pre_loss2_stage3_class += loss_refine_tuple[11].mean().data 
                pre_loss3_stage3_class += loss_refine_tuple[12].mean().data 
                pre_loss4_stage3_class += loss_refine_tuple[13].mean().data 

                pre_loss1_stage3_class_satge2 += loss_refine_tuple[6].mean().data
                pre_loss2_stage3_class_satge2 += loss_refine_tuple[7].mean().data 
                pre_loss3_stage3_class_satge2 += loss_refine_tuple[8].mean().data 
                pre_loss4_stage3_class_satge2 += loss_refine_tuple[9].mean().data 

                pre_loss1_stage3_score += loss_refine_tuple[14].mean().data
                pre_loss2_stage3_score += loss_refine_tuple[15].mean().data 
                pre_loss3_stage3_score += loss_refine_tuple[16].mean().data 
                pre_loss4_stage3_score += loss_refine_tuple[17].mean().data 

                acc = correct_tuple[0].sum()  / (correct_tuple[0].sum() + correct_tuple[1].sum() ) 
                acc_refine = torch.zeros(1)
                if len(correct_refine_tuple) == 4:
                    acc_refine = (correct_refine_tuple[0].sum()  + correct_refine_tuple[1].sum() ) / (correct_refine_tuple[0].sum()  + \
                                        correct_refine_tuple[1].sum() + correct_refine_tuple[2].sum() + correct_refine_tuple[3].sum() )
                data = (loss, acc, acc_refine, loss_tuple, loss_refine_tuple)
                utils.add_log_batch(logger, data, batch_idx + epoch * len(dataloader), mode=mode, method="refine")
                
                if use_log:
                    record_stage2 = utils.eval_and_log(logger, data_path, keep_grasp_num_stage2, stage2_mask, grasp_stage2, \
                                            record_stage2, self.eval_params, batch_idx + epoch*len(dataloader), mode, stage='stage2')

                    record_stage3 = utils.eval_and_log(logger, data_path, keep_grasp_num_stage3, stage3_mask, select_grasp_class, \
                                            record_stage3, self.eval_params, batch_idx + epoch*len(dataloader), mode, stage='stage3_class')
                
                    record_stage3_stage2 = utils.eval_and_log(logger, data_path, keep_grasp_num_stage3, stage3_mask, select_grasp_class_stage2, \
                                    record_stage3_stage2, self.eval_params, batch_idx + epoch*len(dataloader), mode, stage='stage3_class_stage2')
                
                    record_stage3_score = utils.eval_and_log(logger, data_path, keep_grasp_num_stage3_score, stage3_score_mask, select_grasp_score, \
                                    record_stage3_score, self.eval_params, batch_idx + epoch*len(dataloader), mode, stage='stage3_score')

                print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(mode,
                            epoch, batch_idx * batch_size, len(dataloader.dataset),
                            100. * batch_idx * batch_size / len(dataloader.dataset), (loss_total).item(), args.tag))
            except:
                loss_total = loss.sum() 
                if mode == 'train':
                    loss_total.backward()
                    optimizer_score.step()
                    optimizer_region.step()

        data = (pre_loss1_stage2/batch_idx, pre_loss2_stage2/batch_idx, pre_loss3_stage2/batch_idx, pre_loss4_stage2/batch_idx, \
                pre_loss1_stage3_class_satge2/batch_idx, pre_loss2_stage3_class_satge2/batch_idx, pre_loss3_stage3_class_satge2/batch_idx, pre_loss4_stage3_class_satge2/batch_idx, \
                pre_loss1_stage3_class/batch_idx, pre_loss2_stage3_class/batch_idx, pre_loss3_stage3_class/batch_idx, pre_loss4_stage3_class/batch_idx, \
                pre_loss1_stage3_score/batch_idx, pre_loss2_stage3_score/batch_idx, pre_loss3_stage3_score/batch_idx, pre_loss4_stage3_score/batch_idx)
        utils.add_log_epoch(logger, data, epoch, mode=mode, method="refine")
        if use_log:
            records = [record_stage2, record_stage3, record_stage3_stage2, record_stage3_score]
            stages  = ['stage2', 'stage3_class', 'stage3_class_stage2','stage3_score']
            utils.add_eval_log_epoch(logger, records, len(dataloader.dataset), epoch, mode, stages)

        if mode == 'train':
            scheduler_score.step()
            scheduler_region.step()

    def train_refine(self, epoch):
        self.train_val(epoch, mode='train', use_log=False)

    def validate_refine(self, epoch):
        self.train_val(epoch, mode='validate', use_log=True)
 
    def test_refine(self, epoch, mode):
        self.train_val(epoch, mode=mode, use_log=True)
               
    def train(self):
        for epoch in range(self.start_epoch, self.end_epoch):
            print("---------------train_refine epoch", epoch, "------------------")
            path_score  = os.path.join(self.saved_base_path, 'score_{}.model'.format(epoch))
            path_region = os.path.join(self.saved_base_path, 'region_{}.model'.format(epoch))
            
            self.train_refine(epoch)
            torch.save(score_model, path_score)
            torch.save(region_model, path_region)
            self.validate_refine(epoch)
            # self.test_refine(epoch, mode='test')

    def validate(self, epoch):
        print("---------------validate_score epoch", epoch, "------------------")
        self.validate_refine(epoch)

    def test(self, epoch):
        print("---------------DATALOADER: test_score epoch", epoch, "------------------")
        self.test_refine(epoch, mode='test_single')

def main():
    if args.mode == 'pretrain_score':
        scoreModule = ScoreModule(resume_epoch, args.epoch, train_score_loader, val_score_loader, test_score_loader)
        scoreModule.train()

    elif args.mode == 'pretrain_region':
        regionModule = RegionModule(resume_epoch, args.epoch, train_score_loader, val_score_loader, test_score_loader)
        regionModule.train()
            
    elif args.mode == 'train':
        refineModule = RefineModule(resume_epoch, args.epoch, train_score_loader, val_score_loader, test_score_loader)
        refineModule.train()

    elif args.mode == 'validate_score':
        scoreModule = ScoreModule(val_data_loader=val_score_loader)
        scoreModule.validate(resume_epoch)

    elif args.mode == 'validate_region':
        regionModule = RegionModule(val_data_loader=val_score_loader)
        regionModule.validate(resume_epoch)

    elif args.mode == 'validate':
        refineModule = RefineModule(val_data_loader=val_score_loader)
        refineModule.validate(resume_epoch)

    elif args.mode == 'test_score':
        scoreModule = ScoreModule(test_data_loader=test_score_loader)
        scoreModule.test(resume_epoch)

    elif args.mode == 'test_region':
        regionModule = RegionModule(test_data_loader=test_score_loader)
        regionModule.test(resume_epoch)

    elif args.mode == 'test':
        refineModule = RefineModule(test_data_loader=test_score_loader)
        refineModule.test(resume_epoch)

if __name__ == "__main__":
    main()
