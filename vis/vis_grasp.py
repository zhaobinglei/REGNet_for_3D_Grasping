import numpy as np
import open3d, os
import torch
from visualization_utils import get_hand_geometry
import transforms3d
torch.set_printoptions(precision=8)

def inv_transform_grasp(grasp_trans):
    '''
      Input:
        grasp_trans:[B, 8] (center[3], axis_y[3], grasp_angle[1], score[1])
      Output:
        matrix: [B, 3, 4] 
                [[x1, y1, z1, c1],
                 [x2, y2, z2, c2],
                 [x3, y3, z3, c3]]
        grasp_score_ori: [B]
    '''
    grasp_trans = torch.Tensor(grasp_trans).float()
    no_grasp_mask = (grasp_trans.view(-1,8)[:,-1] == -1)

    center = grasp_trans.view(-1,8)[:,:3]
    axis_y = grasp_trans.view(-1,8)[:,3:6]
    angle = grasp_trans.view(-1,8)[:,6]
    cos_t, sin_t = torch.cos(angle), torch.sin(angle)

    B = len(grasp_trans.view(-1,8))
    R1 = torch.zeros((B, 3, 3))
    for i in range(B):
        r = torch.tensor([[cos_t[i], 0, -sin_t[i]],[0, 1, 0],[sin_t[i], 0, cos_t[i]]]).view(1,3,3)
        R1[i,:,:] = r

    norm_y = torch.norm(axis_y, dim=1)
    axis_y = torch.div(axis_y, norm_y.view(-1,1))
    zero = torch.zeros((B, 1), dtype=torch.float32)
    if axis_y.is_cuda:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float).cuda()
        zero = zero.cuda()
        R1 = R1.cuda()
    else:
        axis_y[torch.nonzero(torch.eq(norm_y, 0))] = torch.tensor(([0,1,0]), dtype=torch.float)
    axis_x = torch.cat((axis_y[:, 1].view(-1,1), -axis_y[:, 0].view(-1,1), zero), 1)
    norm_x = torch.norm(axis_x, dim=1)
    axis_x = torch.div(axis_x, norm_x.view(-1,1))
    if axis_y.is_cuda:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        axis_x[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

    axis_z = torch.cross(axis_x, axis_y, dim=1)
    norm_z = torch.norm(axis_z, dim=1)
    axis_z = torch.div(axis_z, norm_z.view(-1,1))
    if axis_z.is_cuda:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float).cuda()
    else:
        axis_z[torch.nonzero(torch.eq(norm_z, 0))] = torch.tensor(([0,0,1]), dtype=torch.float)
    matrix = torch.cat((axis_x.view(-1,3,1), axis_y.view(-1,3,1), axis_z.view(-1,3,1)), dim=2)
    matrix = torch.bmm(matrix, R1)
    approach = matrix[:,:,0]
    norm_x = torch.norm(approach, dim=1)
    approach = torch.div(approach, norm_x.view(-1,1))
    if approach.is_cuda:
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float).cuda()
    else:
        approach[torch.nonzero(torch.eq(norm_x, 0))] = torch.tensor(([1,0,0]), dtype=torch.float)

    minor_normal = torch.cross(approach, axis_y, dim=1)
    matrix = torch.cat((approach.view(-1,3,1), axis_y.view(-1,3,1), minor_normal.view(-1,3,1), center.view(-1,3,1)), dim=2)#.permute(0,2,1)
    matrix[no_grasp_mask] = -1
    matrix = matrix.view(len(grasp_trans), -1, 3, 4)
    

    grasp_score_ori = grasp_trans[:,7]
    grasp_score_ori = grasp_score_ori.view(-1)
    grasp_score_ori[no_grasp_mask] = -1
    grasp_score_ori = grasp_score_ori.view(len(grasp_trans), -1)

    return matrix.numpy(), grasp_score_ori.numpy()

def draw_one_grasp(grasp, color_list=[0, 0.5, 0]):
    grasp = np.r_[grasp.reshape(3,4), np.array([[0,0,0,1]])]

    global_to_local = np.linalg.inv(grasp)
    hand = get_hand_geometry(global_to_local, color=color_list)
    return hand

def show_grasp(path, stage: str, score_thre: float):
    data = np.load(os.path.abspath(path), allow_pickle=True)
    view = data['points']
    color = data['colors']
    view_point_cloud = open3d.geometry.PointCloud()
    view_point_cloud.points = open3d.utility.Vector3dVector(view)
    view_point_cloud.colors = open3d.utility.Vector3dVector(color)

    grasp = data[stage]
    grasp, score = inv_transform_grasp(grasp)
    print(grasp.shape)
    
    vis_list = [view_point_cloud]
    score_max, score_max_idx = 0, 0
    show_idxs = []
    for idx in range(len(grasp)):
        if score[idx] < 0.55:
            continue
        show_idxs.append(idx)
        if score_max < score[idx]:
            score_max = score[idx]
            score_max_idx = idx

    show_idxs.remove(score_max_idx)
    for idx in show_idxs:
        i = grasp[idx]
        hand = draw_one_grasp(i)
        vis_list.extend(hand)
    hand = draw_one_grasp(grasp[score_max_idx], color_list=[0.5, 0, 0])
    vis_list.extend(hand)

    open3d.visualization.draw_geometries(vis_list)

if __name__ == '__main__':
    path = "test_file/virtual_data_predict/00001_view_1.p"
    stage = 'grasp_stage2'
    score_thre = 0.5
    show_grasp(path, stage, score_thre)
    