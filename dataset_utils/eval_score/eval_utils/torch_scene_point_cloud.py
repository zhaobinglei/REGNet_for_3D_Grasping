import open3d
import numpy as np
import torch
import sys
from .pointcloud import PointCloud

class TorchScenePointCloud(PointCloud):
    def __init__(self, data, voxelize=True, visualization=False):
        points = data['scene_cloud']
        cloud = open3d.geometry.PointCloud()
        cloud.points = open3d.utility.Vector3dVector(points)

        if 'scene_normal' in data.keys():
            normals = data['scene_normal']
            cloud.normals = open3d.utility.Vector3dVector(normals)
            PointCloud.__init__(self, cloud, visualization=visualization)
        else:
            PointCloud.__init__(self, cloud, visualization=visualization)
            self.estimate_normals()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cloud_array_homo = torch.cat(
            [torch.tensor(points.T).float(), torch.ones(1, points.shape[0])], dim=0).float().to(device)    
        self.normal_array = torch.tensor(self.normals.T).float().to(device)
        self.kd_tree = open3d.geometry.KDTreeFlann(self.cloud)

if __name__ == '__main__':
    exit(0)
    data_path = '/data/zbl/s4g/eval_data/0_view_0_noise.p'
    data = np.load(data_path, allow_pickle=True)
    pc = TorchScenePointCloud(data)
