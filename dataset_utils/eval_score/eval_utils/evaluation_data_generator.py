# MIT License

# Copyright (c) 2021 yzqin 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import open3d, transforms3d
import numpy as np
from tqdm import tqdm, trange
from ..configs import config
import os
import torch

from .pointcloud import PointCloud
from .torch_scene_point_cloud import TorchScenePointCloud


CAMERA_POSE = [
    [0.8, 0, 1.7, 0.948, 0, 0.317, 0],
    [-0.8, 0, 1.6, -0.94, 0, 0.342, 0],
    [0.0, 0.75, 1.7, 0.671, -0.224, 0.224, 0.671],
    [0.0, -0.75, 1.6, -0.658, -0.259, -0.259, 0.658]
]

def transformation_quat(point):
    T_local_to_global = np.eye(4)
    quat = transforms3d.quaternions.axangle2quat([1,0,0], np.pi*1.13)
    return np.r_[point, quat]

class EvalDataTest(PointCloud):
    def __init__(self, points, grasp, view_num, table_height,
                 depth: float, width: float, gpu: int, visualization=False):
        '''
          points: [N, 3]
          grasp : [B, 8]
        '''
        if gpu != -1:
            torch.cuda.set_device(gpu)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        PointCloud.__init__(self, cloud, visualization)
        self.table_height = table_height
        
        if view_num is None:
            center_camera = np.array([0, 0, 1.658])
        else:
            center_camera = CAMERA_POSE[view_num][0:3]

        self.depth, self.width = depth, width

        if type(points) == torch.Tensor:
            self.cloud_array = points.float().to(self.device)
        else:
            self.cloud_array = torch.FloatTensor(points).to(self.device)
        self.cloud_array_homo = torch.cat(
            [self.cloud_array.transpose(0, 1), torch.ones(1, self.cloud_array.shape[0], device=self.device)],
            dim=0).float().to(self.device)
            
        self.estimate_normals(center_camera) 
        self.normal_array = torch.tensor(self.normals.T).float().to(self.device)
        
        if type(grasp) == torch.Tensor:
            self.grasp_no_collision_view = grasp.float().to(self.device)
        else:
            self.grasp_no_collision_view = torch.FloatTensor(grasp).to(self.device)

        self.frame, self.center, self.score = self.inv_transform_predicted_grasp(self.grasp_no_collision_view)
        #### torch.Tensor: self.frame [B,3,3], self.center [B,3], self.score [B,1]

        self.global_to_local = torch.eye(4).unsqueeze(0).expand(self.frame.shape[0], 4, 4).to(self.device).contiguous()
        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2), self.center.unsqueeze(2))
        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

        # self.baseline_frame = torch.zeros(self.frame.shape[0], 4, 4, dtype=torch.float32, device=self.device)
        self.baseline_frame_index = torch.zeros(self.frame.shape[0], dtype=torch.float32, device=self.device)
        self.valid_grasp = 0

    def run_collision_view(self):
        #print('\n Start test view collision checking \n')
        for frame_index in range(self.frame.shape[0]):
            self.finger_hand_view(frame_index)
        #print('\n Finish view collision checking \n')
        return self.grasp_no_collision_view[self.baseline_frame_index[:self.valid_grasp].long()]     

    def inv_transform_predicted_grasp(self, grasp_trans):
        '''
          Input:
            grasp_trans:[B, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
          Output:
            matrix: [B, 3, 3] 
                [[x1, y1, z1],
                 [x2, y2, z2],
                 [x3, y3, z3]]
            center: [B, 3]
                [c1, c2, c3]
            score: [B, 1]
        '''
        grasp_trans = grasp_trans.view(-1,8)

        center = grasp_trans[:,:3].contiguous()
        axis_y = grasp_trans[:,3:6]
        angle = grasp_trans[:,6]
        cos_t, sin_t = torch.cos(angle), torch.sin(angle)

        B = len(grasp_trans)
        # R1 = torch.zeros((B, 3, 3))
        # for i in range(B):
        #     r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
        #     R1[i,:,:] = r

        one, zero = torch.ones((B, 1), dtype=torch.float32).to(self.device), torch.zeros((B, 1), dtype=torch.float32).to(self.device)
        R1 = torch.cat( (cos_t.view(B,1), zero, -sin_t.view(B,1), zero, one, zero, sin_t.view(B,1), 
                                        zero, cos_t.view(B,1)), dim=1).view(B,3,3).to(self.device)
        
        norm_y = torch.norm(axis_y, dim=1)
        axis_y = torch.div(axis_y, norm_y.view(-1,1))
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).to(self.device)
            
        axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
        norm_x = torch.norm(axis_x, dim=1)
        axis_x = torch.div(axis_x, norm_x.view(-1,1))
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(self.device)

        axis_z = torch.cross(axis_x, axis_y, dim=1)
        norm_z = torch.norm(axis_z, dim=1)
        axis_z = torch.div(axis_z, norm_z.view(-1,1))
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).to(self.device)
        
        matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
        matrix = torch.bmm(matrix, R1)
        approach = matrix[:,:,0]
        norm_x = torch.norm(approach, dim=1)
        approach = torch.div(approach, norm_x.view(-1,1))
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(self.device)

        minor_normal = torch.cross(approach, axis_y, dim=1)
        matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1)), dim=2).contiguous()

        score = grasp_trans[:,7].view(-1, 1).contiguous()
        return matrix, center, score

    def _table_collision_check(self, point, frame):
        """
        Check whether the gripper collide with the table top with offset
        :param point: torch.tensor(3)
        :param frame: torch.tensor(3, 3)
        """
        T_local_to_global = torch.eye(4, device=self.device).float()
        T_local_to_global[0:3, 0:3] = frame
        T_local_to_global[0:3, 3] = point
        T_local_search_to_global = T_local_to_global.squeeze(0).expand(1, 4, 4).contiguous()
        config_gripper = torch.tensor(np.array(config.TORCH_GRIPPER_BOUND.squeeze(0).expand(1, -1, -1).contiguous())).to(self.device)
        boundary_global = torch.bmm(T_local_search_to_global, config_gripper)
        table_collision_bool = boundary_global[:, 2, :] < self.table_height - 0.005 #+ config.TABLE_COLLISION_OFFSET
        return table_collision_bool.any(dim=1, keepdim=False)

    def finger_hand_view(self, frame_index):
        """
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        """
        frame = self.frame[frame_index, :, :]
        point = self.center[frame_index, :]
        
        if point[2] + frame[2, 0] * self.depth < self.table_height - 0.005: # config.FINGER_LENGTH  self.depth 
            return

        table_collision_bool = self._table_collision_check(point, frame)

        T_global_to_local = self.global_to_local[frame_index, :, :]
        local_cloud = torch.matmul(T_global_to_local, self.cloud_array_homo)
        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], self.normal_array)

        close_plane_bool = (local_cloud[0, :] > - config.BOTTOM_LENGTH) & (local_cloud[0, :] < self.depth) # config.FINGER_LENGTH
        if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
            return
        local_search_close_plane_points = local_cloud[:, close_plane_bool][0:3, :]  # only filter along x axis
        #T_local_to_local_search = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], 
        #                                        [0., 0., 1., 0.], [0., 0., 0., 1.]])
        #local_search_close_plane_points = torch.matmul(T_local_to_local_search.contiguous().view(-1, 4), \
        #                                               close_plane_points).contiguous().view(1, 4, -1)[:, 0:3, :]

        hand_half_bottom_width = self.width / 2 + config.FINGER_WIDTH
        hand_half_bottom_space = self.width / 2 
        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                            (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)
        back_collision_bool = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                                z_collision_bool

        if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
            return

        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] > hand_half_bottom_space)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] < -hand_half_bottom_space)

        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)
        if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
            return
            
        # self.baseline_frame[self.valid_grasp] = self.global_to_local[frame_index]
        self.baseline_frame_index[self.valid_grasp] = frame_index
        self.valid_grasp += 1


class EvalDataValidate(PointCloud):
    def __init__(self, data, grasp, view_num, table_height,
                 depth, width: float, gpu: int, visualization=False):
        '''
          data:  dict {'point_cloud': [3,N1], 'view_cloud' : [N1,3],
                     'scene_cloud': [N2,3], 'scene_cloud_table': [N3,3], ... }
          grasp: [B, 8] or [B, 4, 4]
        '''
        if gpu != -1:
            torch.cuda.set_device(gpu)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        points = data['view_cloud']
        self.table_height = table_height

        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)
        PointCloud.__init__(self, cloud, visualization)
        camera_pose = CAMERA_POSE[view_num][0:3]
        self.depth, self.width = depth, width

        if type(points) == torch.Tensor:
            self.cloud_array = points.float().to(self.device)
        else:
            self.cloud_array = torch.FloatTensor(points).to(self.device)
        self.cloud_array_homo = torch.cat(
            [self.cloud_array.transpose(0, 1), torch.ones(1, self.cloud_array.shape[0], device=self.device)],
            dim=0).float().to(self.device)
        
        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)
        self.estimate_normals(camera_pose)
        self.normal_array = torch.tensor(self.normals.T).float().to(self.device) 

        if type(grasp) == torch.Tensor:
            self.grasp_no_collision_view = grasp.float().to(self.device)
        else:
            self.grasp_no_collision_view = torch.FloatTensor(grasp).to(self.device)

        if self.grasp_no_collision_view.shape[1] == 8:
            self.frame, self.center, self.score = self.inv_transform_predicted_grasp(self.grasp_no_collision_view)
        else:
            self.frame, self.center, self.score = self.grasp_no_collision_view[:,:3,:3].contiguous(),\
                self.grasp_no_collision_view[:,:3,3].contiguous(), self.grasp_no_collision_view[:,:3,3].contiguous()
        #### self.frame [B,3,3], self.center [B,3], self.score [B,1]

        self.global_to_local = torch.eye(4).unsqueeze(0).expand(self.frame.shape[0], 4, 4).to(self.device).contiguous()
        self.global_to_local[:, 0:3, 0:3] = self.frame.transpose(1, 2)
        self.global_to_local[:, 0:3, 3:4] = -torch.bmm(self.frame.transpose(1, 2), self.center.unsqueeze(2))

        self.baseline_frame = torch.zeros(self.frame.shape[0], 4, 4, dtype=torch.float32, device=self.device)
        self.baseline_frame_index = torch.zeros(self.frame.shape[0], dtype=torch.float32, device=self.device)
        self.valid_grasp = 0
        
        self.antipodal_score = None
        self.collision_bool = None
        self.label_bool = None

        self.left_normal = torch.tensor([[0, 1, 0]], device=self.device).float()
        self.right_normal = torch.tensor([[0, -1, 0]], device=self.device).float()

        self.scene = TorchScenePointCloud(data)
       
    def inv_transform_predicted_grasp(self, grasp_trans):
        '''
          Input:
            grasp_trans:[B, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
          Output:
            matrix: [B, 3, 3] 
                [[x1, y1, z1],
                 [x2, y2, z2],
                 [x3, y3, z3]]
            center: [B, 3]
                [c1, c2, c3]
            score: [B, 1]
        '''
        grasp_trans = grasp_trans.view(-1,8)

        center = grasp_trans[:,:3].contiguous()
        axis_y = grasp_trans[:,3:6]
        angle = grasp_trans[:,6]
        cos_t, sin_t = torch.cos(angle), torch.sin(angle)

        B = len(grasp_trans)
        # R1 = torch.zeros((B, 3, 3))
        # for i in range(B):
        #     r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
        #     R1[i,:,:] = r

        one, zero = torch.ones((B, 1), dtype=torch.float32).to(self.device), torch.zeros((B, 1), dtype=torch.float32).to(self.device)
        R1 = torch.cat( (cos_t.view(B,1), zero, -sin_t.view(B,1), zero, one, zero, sin_t.view(B,1), 
                                        zero, cos_t.view(B,1)), dim=1).view(B,3,3).to(self.device)
        
        norm_y = torch.norm(axis_y, dim=1)
        axis_y = torch.div(axis_y, norm_y.view(-1,1))
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).to(self.device)
            
        axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
        norm_x = torch.norm(axis_x, dim=1)
        axis_x = torch.div(axis_x, norm_x.view(-1,1))
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(self.device)

        axis_z = torch.cross(axis_x, axis_y, dim=1)
        norm_z = torch.norm(axis_z, dim=1)
        axis_z = torch.div(axis_z, norm_z.view(-1,1))
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).to(self.device)
        
        matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
        matrix = torch.bmm(matrix, R1)
        approach = matrix[:,:,0]
        norm_x = torch.norm(approach, dim=1)
        approach = torch.div(approach, norm_x.view(-1,1))
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).to(self.device)

        minor_normal = torch.cross(approach, axis_y, dim=1)
        matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1)), dim=2).contiguous()

        score = grasp_trans[:,7].view(-1, 1).contiguous()
        return matrix, center, score


    def run_collision(self):
        self.run_collision_view()
        self.run_collision_scene(self.scene)

        select_grasp = torch.nonzero(self.collision_bool).view(-1)
        self.grasp_no_collision_scene = self.grasp_no_collision_view[select_grasp]

        grasp_formal_num = len(self.frame)
        grasp_no_collision_view_num = self.valid_grasp
        grasp_no_collision_scene_num = len(select_grasp)

        total_vgr = grasp_no_collision_scene_num 
        total_score = torch.sum(self.antipodal_score).item()
        return total_vgr, total_score, grasp_no_collision_view_num, self.grasp_no_collision_view, self.grasp_no_collision_scene

    def run_collision_view(self):
        #print('\n Start validate view collision checking \n')
        for frame_index in range(self.frame.shape[0]):
            self.finger_hand_view(frame_index)
        #print('\n Finish view collision checking \n')
        self.grasp_no_collision_view = self.grasp_no_collision_view[self.baseline_frame_index[:self.valid_grasp].long()] 

    def run_collision_scene(self, scene: TorchScenePointCloud):
        #print('\n Start validate scene collision checking \n')
        self.collision_bool = torch.zeros(self.valid_grasp, dtype=torch.uint8, device=self.device)
        self.antipodal_score = torch.zeros(self.valid_grasp, dtype=torch.float, device=self.device)
        for i in range(self.valid_grasp):
            self.finger_hand_scene(self.baseline_frame[i, :, :], i, scene)
        #print('\n Finish scene collision checking \n')

    def _table_collision_check(self, point, frame):
        """
        Check whether the gripper collide with the table top with offset
        :param point: torch.tensor(3)
        :param frame: torch.tensor(3, 3)
        """
        T_local_to_global = torch.eye(4, device=self.device).float()
        T_local_to_global[0:3, 0:3] = frame
        T_local_to_global[0:3, 3] = point
        T_local_search_to_global = T_local_to_global.squeeze(0).expand(1, 4, 4).contiguous()
        config_gripper = torch.tensor(np.array(config.TORCH_GRIPPER_BOUND.squeeze(0).expand(1, -1, -1).contiguous())).to(self.device)
        boundary_global = torch.bmm(T_local_search_to_global, config_gripper)
        table_collision_bool = boundary_global[:, 2, :] < self.table_height - 0.005 #+ config.TABLE_COLLISION_OFFSET
        return table_collision_bool.any(dim=1, keepdim=False)

    def _antipodal_score(self, close_region_cloud, close_region_cloud_normal):
        """
        Estimate the antipodal score of a single grasp using scene point cloud
        Antipodal score is proportional to the reciprocal of friction angle
        Antipodal score is also divided by the square of objects in the closing region
        :param close_region_cloud: The point cloud in the gripper closing region, torch.tensor (3, n)
        :param close_region_cloud_normal: The point cloud normal in the gripper closing region, torch.tensor (3, n)
        """
        assert close_region_cloud.shape == close_region_cloud_normal.shape, \
            "Points and corresponding normals should have same shape"

        left_y = torch.max(close_region_cloud[1, :])
        right_y = torch.min(close_region_cloud[1, :])
        normal_search_depth = torch.min((left_y - right_y) / 3, config.NEIGHBOR_DEPTH.to(self.device) )

        left_region_bool = close_region_cloud[1, :] > left_y - normal_search_depth
        right_region_bool = close_region_cloud[1, :] < right_y + normal_search_depth
        left_normal_theta = torch.abs(torch.matmul(self.left_normal, close_region_cloud_normal[:, left_region_bool]))
        right_normal_theta = torch.abs(torch.matmul(self.right_normal, close_region_cloud_normal[:, right_region_bool]))

        geometry_average_theta = torch.mean(left_normal_theta) * torch.mean(right_normal_theta)
        return geometry_average_theta

    def finger_hand_view(self, frame_index):
        """
        :param frame_index: The index of point in frame list, which is a subset of all point cloud
        """
        frame = self.frame[frame_index, :, :]
        point = self.center[frame_index, :]
        
        if type(self.depth) is float:
            if point[2] + frame[2, 0] * self.depth < self.table_height - 0.005: # config.FINGER_LENGTH  self.depth 
                return
        else:
            if point[2] + frame[2, 0] * self.depth[frame_index] < self.table_height - 0.005: # config.FINGER_LENGTH  self.depth 
                return

        table_collision_bool = self._table_collision_check(point, frame)

        T_global_to_local = self.global_to_local[frame_index, :, :]
        local_cloud = torch.matmul(T_global_to_local, self.cloud_array_homo)
        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], self.normal_array)

        if type(self.depth) is float:
            close_plane_bool = (local_cloud[0, :] > - config.BOTTOM_LENGTH) & (local_cloud[0, :] < self.depth) # config.FINGER_LENGTH
        else:
            close_plane_bool = (local_cloud[0, :] > - config.BOTTOM_LENGTH) & (local_cloud[0, :] < self.depth[frame_index]) # config.FINGER_LENGTH
        if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
            return
        local_search_close_plane_points = local_cloud[:, close_plane_bool][0:3, :]  # only filter along x axis
        #T_local_to_local_search = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], 
        #                                        [0., 0., 1., 0.], [0., 0., 0., 1.]])
        #local_search_close_plane_points = torch.matmul(T_local_to_local_search.contiguous().view(-1, 4), \
        #                                               close_plane_points).contiguous().view(1, 4, -1)[:, 0:3, :]

        hand_half_bottom_width = self.width / 2 + config.FINGER_WIDTH
        hand_half_bottom_space = self.width / 2 
        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                            (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)
        back_collision_bool = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                                z_collision_bool

        if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
            return

        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] > hand_half_bottom_space)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] < -hand_half_bottom_space)

        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)
        if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
            return

        close_region_bool = z_collision_bool & \
                            (local_search_close_plane_points[1, :] < hand_half_bottom_space) & \
                            (local_search_close_plane_points[1, :] > -hand_half_bottom_space)    
        close_region_point_num = torch.sum(close_region_bool)
        if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
            return

        self.baseline_frame[self.valid_grasp] = self.global_to_local[frame_index]
        self.baseline_frame_index[self.valid_grasp] = frame_index
        self.valid_grasp += 1

    def finger_hand_scene(self, T_global_to_local, valid_index, scene: TorchScenePointCloud):
        """
        Local search one point and store the closing region point num of each configurations
        Search height first, then width, finally theta
        Save the number of points in the close region if the grasp do not fail in local search
        Save the score of antipodal_grasp, note that multi-objects heuristic is also stored here
        """
        local_cloud = torch.matmul(T_global_to_local, scene.cloud_array_homo)
        local_cloud_normal = torch.matmul(T_global_to_local[0:3, 0:3], scene.normal_array)
        if type(self.depth) is float:
            close_plane_bool = (local_cloud[0, :] > - config.BOTTOM_LENGTH) & (local_cloud[0, :] < self.depth) # config.FINGER_LENGTH  
        else:
            close_plane_bool = (local_cloud[0, :] > - config.BOTTOM_LENGTH) & (local_cloud[0, :] < self.depth[self.baseline_frame_index[valid_index].int()]) # config.FINGER_LENGTH
        
                
        if torch.sum(close_plane_bool) < config.NUM_POINTS_THRESHOLD:
            return

        local_search_close_plane_points = local_cloud[:, close_plane_bool][0:3, :]  # only filter along x axis

        hand_half_bottom_width = self.width / 2 + config.FINGER_WIDTH
        hand_half_bottom_space = self.width / 2 
        z_collision_bool = (local_search_close_plane_points[2, :] < config.HALF_HAND_THICKNESS) & \
                           (local_search_close_plane_points[2, :] > -config.HALF_HAND_THICKNESS)
        back_collision_bool = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                              (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                              (local_search_close_plane_points[0, :] < -config.BACK_COLLISION_MARGIN) & \
                              z_collision_bool

        if torch.sum(back_collision_bool) > config.BACK_COLLISION_THRESHOLD:
            return

        y_finger_region_bool_left = (local_search_close_plane_points[1, :] < hand_half_bottom_width) & \
                                    (local_search_close_plane_points[1, :] > hand_half_bottom_space)
        y_finger_region_bool_right = (local_search_close_plane_points[1, :] > -hand_half_bottom_width) & \
                                     (local_search_close_plane_points[1, :] < -hand_half_bottom_space)

        y_finger_region_bool = y_finger_region_bool_left | y_finger_region_bool_right
        collision_region_bool = (z_collision_bool & y_finger_region_bool)
        if torch.sum(collision_region_bool) > config.FINGER_COLLISION_THRESHOLD:
            return
            
        close_region_bool = z_collision_bool & \
                            (local_search_close_plane_points[1, :] < hand_half_bottom_space) & \
                            (local_search_close_plane_points[1, :] > -hand_half_bottom_space)
        close_region_point_num = torch.sum(close_region_bool)
        if close_region_point_num < config.CLOSE_REGION_MIN_POINTS:
            return

        self.collision_bool[valid_index] = 1
        close_region_normals = local_cloud_normal[:, close_plane_bool][:, close_region_bool]
        close_region_cloud = local_search_close_plane_points[:, close_region_bool]

        self.antipodal_score[valid_index] = self._antipodal_score(close_region_cloud, close_region_normals)
