import torch
import torch.nn as nn

from .nn import SharedMLP
from . import function as _F
from .functions.gather_knn import gather_knn

#from tdgpd.functions.functions import gather_knn


class FarthestPointSampler(nn.Module):
    """Farthest point sampler

    Args:
        num_centroids (int): the number of centroids

    """

    def __init__(self, num_centroids):
        super(FarthestPointSampler, self).__init__()
        self.num_centroids = num_centroids

    def forward(self, points):
        with torch.no_grad():
            index = _F.farthest_point_sample(points, self.num_centroids)
        return index

    def extra_repr(self):
        return 'num_centroids={:d}'.format(self.num_centroids)


class QueryGrouper(nn.Module):
    def __init__(self, radius, num_neighbours):
        super(QueryGrouper, self).__init__()
        assert radius > 0.0 and num_neighbours > 0
        self.radius = radius
        self.num_neighbours = num_neighbours

    def forward(self, new_xyz, xyz, feature, use_xyz):
        with torch.no_grad():
            index, unique_count = _F.ball_query(xyz, new_xyz, self.radius, self.num_neighbours)

        # (batch_size, 3, num_centroids, num_neighbours)
        group_xyz = _F.group_points(xyz, index)
        # translation normalization
        group_xyz -= new_xyz.unsqueeze(-1)

        if feature is not None:
            # (batch_size, channels, num_centroids, num_neighbours)
            group_feature = _F.group_points(feature, index)
            if use_xyz:
                group_feature = torch.cat([group_xyz, group_feature], dim=1)
        else:
            group_feature = group_xyz

        return group_feature, group_xyz

    def extra_repr(self):
        return 'radius={}, num_neighbours={}'.format(self.radius, self.num_neighbours)


# ---------------------------------------------------------------------------- #
# EdgeConv
# ---------------------------------------------------------------------------- #
class EdgeQueryGrouper(nn.Module):
    def __init__(self, radius, num_neighbours):
        super(EdgeQueryGrouper, self).__init__()
        assert radius > 0.0 and num_neighbours > 0
        self.radius = radius
        self.num_neighbours = num_neighbours

    def forward(self, new_xyz, xyz, centroid_feature, feature, use_xyz):
        with torch.no_grad():
            index, unique_count = _F.ball_query(xyz, new_xyz, self.radius, self.num_neighbours)

        # (batch_size, 3, num_centroids, num_neighbours)
        group_xyz = _F.group_points(xyz, index)
        # translation normalization
        group_xyz -= new_xyz.unsqueeze(-1)

        if feature is not None:  # (batch_size, channels, num_centroids)
            # (batch_size, channels, num_centroids, num_neighbours)
            group_feature = _F.group_points(feature, index)
            group_feature2 = group_feature - centroid_feature.unsqueeze(-1)
            if use_xyz:
                group_feature = torch.cat([group_xyz, group_feature, group_feature2], dim=1)
            else:
                group_feature = torch.cat([group_feature, group_feature2], dim=1)
        else:
            group_feature = group_xyz

        return group_feature, group_xyz

    def extra_repr(self):
        return 'radius={}, num_neighbours={}'.format(self.radius, self.num_neighbours)


class FeatureInterpolator(nn.Module):
    def __init__(self, num_neighbors, eps=1e-10):
        super(FeatureInterpolator, self).__init__()
        self.num_neighbors = num_neighbors
        self._eps = eps

    def forward(self, dense_xyz, sparse_xyz, dense_feature, sparse_feature):
        """

        Args:
            dense_xyz: query xyz, (B, 3, N1)
            sparse_xyz: key xyz, (B, 3, N2)
            dense_feature: (B, C1, N1), feature corresponding to xyz1
            sparse_feature: (B, C2, N2), feature corresponding to xyz2

        Returns:
            new_feature: (B, C1+C2, N1), propagated feature

        """
        with torch.no_grad():
            # index: (B, N1, K), distance: (B, N1, K)
            index, distance = _F.search_nn_distance(dense_xyz, sparse_xyz, self.num_neighbors)
            inv_distance = 1.0 / torch.clamp(distance, min=self._eps)
            norm = torch.sum(inv_distance, dim=2, keepdim=True)
            weight = inv_distance / norm

        interpolated_feature = _F.feature_interpolate(sparse_feature, index, weight)

        if dense_feature is not None:
            new_feature = torch.cat([interpolated_feature, dense_feature], dim=1)
        else:
            new_feature = interpolated_feature

        return new_feature

    def extra_repr(self):
        return 'num_neighbours={:d}, eps={}'.format(self.num_neighbors, self._eps)


class EdgeFeatureInterpolator(nn.Module):
    def __init__(self, num_neighbors, eps=1e-10):
        super(EdgeFeatureInterpolator, self).__init__()
        self.num_neighbors = num_neighbors
        self._eps = eps

    def forward(self, dense_xyz, sparse_xyz, dense_feature, sparse_feature):
        """

        Args:
            dense_xyz: query xyz, (B, 3, N1)
            sparse_xyz: key xyz, (B, 3, N2)
            dense_feature: (B, C1, N1), feature corresponding to xyz1
            sparse_feature: (B, C2, N2), feature corresponding to xyz2

        Returns:
            new_feature: (B, C1+C2, N1, K), propagated feature

        """
        with torch.no_grad():
            index, distance = _F.search_nn_distance(dense_xyz, sparse_xyz, self.num_neighbors)
            inv_distance = 1.0 / torch.clamp(distance, min=self._eps)
            norm = torch.sum(inv_distance, dim=2, keepdim=True)
            weight = inv_distance / norm
            gathered_feature = gather_knn(sparse_feature, index)  # TODO: ERROR WITH GRADIENTS

        interpolated_feature = _F.feature_interpolate(sparse_feature, index, weight)
        interpolated_feature = interpolated_feature.unsqueeze(-1).expand(-1, -1, -1, self.num_neighbors)
        gathered_feature = torch.cat([interpolated_feature, gathered_feature - interpolated_feature], dim=1)

        if dense_feature is not None:
            dense_feature = dense_feature.unsqueeze(-1).expand(-1, -1, -1, self.num_neighbors)
            new_feature = torch.cat([gathered_feature, dense_feature], dim=1)
        else:
            new_feature = gathered_feature

        return new_feature


class PointNetSAModule(nn.Module):
    """PointNet set abstraction module"""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_centroids,
                 radius,
                 num_neighbours,
                 use_xyz):
        super(PointNetSAModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.num_centroids = num_centroids
        # self.num_neighbours = num_neighbours
        self.use_xyz = use_xyz

        if self.use_xyz:
            in_channels += 3
        self.mlp = SharedMLP(in_channels, mlp_channels, ndim=2, bn=True)

        if num_centroids <= 0:
            self.sampler = None
        else:
            self.sampler = FarthestPointSampler(num_centroids)

        if num_neighbours < 0:
            assert radius < 0.0
            self.grouper = None
        else:
            assert num_neighbours > 0 and radius > 0.0
            self.grouper = QueryGrouper(radius, num_neighbours)

    def forward(self, xyz, feature=None):
        """

        Args:
            xyz (torch.Tensor): (batch_size, 3, num_points)
                xyz coordinates of feature
            feature (torch.Tensor, optional): (batch_size, in_channels, num_points)

        Returns:
            new_xyz (torch.Tensor): (batch_size, 3, num_centroids)
            new_feature (torch.Tensor): (batch_size, out_channels, num_centroids)

        """

        if self.num_centroids == 0:
            # use the origin as the centroid
            new_xyz = xyz.new_zeros(xyz.size(0), 3, 1)  # (batch_size, 3, 1)
            assert self.grouper is None
            group_feature = feature.unsqueeze(2)  # (batch_size, in_channels, 1, num_points)
            group_xyz = xyz.unsqueeze(2)  # (batch_size, 3, 1, num_points)
            if self.use_xyz:
                group_feature = torch.cat([group_xyz, group_feature], dim=1)
        else:
            if self.num_centroids == -1:
                # use all points
                new_xyz = xyz
            else:
                # sample new points
                index = self.sampler(xyz)
                new_xyz = _F.gather_points(xyz, index)  # (batch_size, 3, num_centroids)

            # group_feature, (batch_size, in_channels, num_centroids, num_neighbours)
            group_feature, group_xyz = self.grouper(new_xyz, xyz, feature, use_xyz=self.use_xyz)

        new_feature = self.mlp(group_feature)
        new_feature, _ = torch.max(new_feature, 3)
        return new_xyz, new_feature

    def init_weights(self, init_fn=None):
        self.mlp.init_weights(init_fn)

    def extra_repr(self):
        return 'num_centroids={:d}, use_xyz={}'.format(self.num_centroids, self.use_xyz)


class PointNetSAAvgModule(nn.Module):
    """PointNet set abstraction module"""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_centroids,
                 radius,
                 num_neighbours,
                 use_xyz):
        super(PointNetSAAvgModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.num_centroids = num_centroids
        # self.num_neighbours = num_neighbours
        self.use_xyz = use_xyz

        if self.use_xyz:
            in_channels += 3
        self.mlp = SharedMLP(in_channels, mlp_channels, ndim=2, bn=True)

        if num_centroids <= 0:
            self.sampler = None
        else:
            self.sampler = FarthestPointSampler(num_centroids)

        if num_neighbours < 0:
            assert radius < 0.0
            self.grouper = None
        else:
            assert num_neighbours > 0 and radius > 0.0
            self.grouper = QueryGrouper(radius, num_neighbours)

    def forward(self, xyz, feature=None):
        """

        Args:
            xyz (torch.Tensor): (batch_size, 3, num_points)
                xyz coordinates of feature
            feature (torch.Tensor, optional): (batch_size, in_channels, num_points)

        Returns:
            new_xyz (torch.Tensor): (batch_size, 3, num_centroids)
            new_feature (torch.Tensor): (batch_size, out_channels, num_centroids)

        """

        if self.num_centroids == 0:
            # use the origin as the centroid
            new_xyz = xyz.new_zeros(xyz.size(0), 3, 1)  # (batch_size, 3, 1)
            assert self.grouper is None
            group_feature = feature.unsqueeze(2)  # (batch_size, in_channels, 1, num_points)
            group_xyz = xyz.unsqueeze(2)  # (batch_size, 3, 1, num_points)
            if self.use_xyz:
                group_feature = torch.cat([group_xyz, group_feature], dim=1)
        else:
            if self.num_centroids == -1:
                # use all points
                new_xyz = xyz
            else:
                # sample new points
                index = self.sampler(xyz)
                new_xyz = _F.gather_points(xyz, index)  # (batch_size, 3, num_centroids)

            # group_feature, (batch_size, in_channels, num_centroids, num_neighbours)
            group_feature, group_xyz = self.grouper(new_xyz, xyz, feature, use_xyz=self.use_xyz)

        new_feature = self.mlp(group_feature)
        new_feature = torch.mean(new_feature, 3)
        return new_xyz, new_feature

    def init_weights(self, init_fn=None):
        self.mlp.init_weights(init_fn)

    def extra_repr(self):
        return 'num_centroids={:d}, use_xyz={}'.format(self.num_centroids, self.use_xyz)


class PointNetSAModuleMSG(nn.Module):
    """PointNet set abstraction module (multi scale grouping)"""

    def __init__(self,
                 in_channels,
                 mlp_channels_list,
                 num_centroids,
                 radius_list,
                 num_neighbours_list,
                 use_xyz):
        super(PointNetSAModuleMSG, self).__init__()

        self.in_channels = in_channels
        self.out_channels = sum(mlp_channels[-1] for mlp_channels in mlp_channels_list)
        self.num_centroids = num_centroids
        self.use_xyz = use_xyz

        num_scales = len(mlp_channels_list)
        assert len(radius_list) == num_scales
        assert len(num_neighbours_list) == num_scales

        self.mlp = nn.ModuleList()
        if num_centroids == -1:
            self.sampler = None
        else:
            assert num_centroids > 0
            self.sampler = FarthestPointSampler(num_centroids)
        self.grouper = nn.ModuleList()

        if self.use_xyz:
            in_channels += 3
        for ind in range(num_scales):
            self.mlp.append(SharedMLP(in_channels, mlp_channels_list[ind], ndim=2, bn=True))
            self.grouper.append(QueryGrouper(radius_list[ind], num_neighbours_list[ind]))

    def forward(self, xyz, feature=None):
        """

        Args:
            xyz (torch.Tensor): (batch_size, 3, num_points)
                xyz coordinates of feature
            feature (torch.Tensor, optional): (batch_size, in_channels, num_points)

        Returns:
            new_xyz (torch.Tensor): (batch_size, 3, num_centroids)
            new_feature (torch.Tensor): (batch_size, out_channels, num_centroids)

        """
        if self.num_centroids > 0:
            # sample new points
            index = self.sampler(xyz)
            # (batch_size, 3, num_centroids)
            new_xyz = _F.gather_points(xyz, index)
        else:
            new_xyz = xyz

        # multi-scale
        new_feature_list = []
        for mlp, grouper in zip(self.mlp, self.grouper):
            # (batch_size, in_channels, num_centroids, num_neighbours)
            group_feature, group_xyz = grouper(new_xyz, xyz, feature, use_xyz=self.use_xyz)
            new_feature = mlp(group_feature)
            new_feature, _ = torch.max(new_feature, 3)
            new_feature_list.append(new_feature)

        return new_xyz, torch.cat(new_feature_list, dim=1)

    def init_weights(self, init_fn=None):
        for mlp in self.mlp:
            mlp.init_weights(init_fn)

    def extra_repr(self):
        return 'num_centroids={:d}, use_xyz={}'.format(self.num_centroids, self.use_xyz)


class EdgeSAModule(nn.Module):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_centroids,
                 radius,
                 num_neighbours,
                 use_xyz):
        super(EdgeSAModule, self).__init__()

        if num_centroids != 0:
            in_channels *= 2
        self.out_channels = mlp_channels[-1]
        self.num_centroids = num_centroids
        self.use_xyz = use_xyz

        if self.use_xyz:
            in_channels += 3
        self.mlp = SharedMLP(in_channels, mlp_channels, ndim=2, bn=True)

        if num_centroids <= 0:
            self.sampler = None
        else:
            self.sampler = FarthestPointSampler(num_centroids)

        if num_neighbours < 0:
            assert radius < 0.0
            self.grouper = None
        else:
            assert num_neighbours > 0 and radius > 0.0
            self.grouper = EdgeQueryGrouper(radius, num_neighbours)

    def forward(self, xyz, feature=None):
        """
                Args:
                    xyz (torch.Tensor): (batch_size, 3, num_points)
                        xyz coordinates of feature
                    feature (torch.Tensor, optional): (batch_size, in_channels, num_points)
                Returns:
                    new_xyz (torch.Tensor): (batch_size, 3, num_centroids)
                    new_feature (torch.Tensor): (batch_size, out_channels, num_centroids)
                """

        if self.num_centroids == 0:
            # use the origin as the centroid
            new_xyz = xyz.new_zeros(xyz.size(0), 3, 1)  # (batch_size, 3, 1)
            assert self.grouper is None
            group_feature = feature.unsqueeze(2)  # (batch_size, in_channels, 1, num_points)
            group_xyz = xyz.unsqueeze(2)  # (batch_size, 3, 1, num_points)
            if self.use_xyz:
                group_feature = torch.cat([group_xyz, group_feature], dim=1)
        else:
            if self.num_centroids == -1:
                # use all points
                new_xyz = xyz
                centroid_feature = feature if feature is not None else None
            else:
                # sample new points
                index = self.sampler(xyz)
                new_xyz = _F.gather_points(xyz, index)  # (batch_size, 3, num_centroids)
                # (batch_size, feature_channels, num_centroids)
                centroid_feature = _F.gather_points(feature, index) if feature is not None else None

            # group_feature, (batch_size, in_channels, num_centroids, num_neighbours)
            group_feature, group_xyz = self.grouper(new_xyz, xyz, centroid_feature, feature, use_xyz=self.use_xyz)

        new_feature = self.mlp(group_feature)
        new_feature, _ = torch.max(new_feature, 3)
        return new_xyz, new_feature


class PointnetFPModule(nn.Module):
    """PointNet feature propagation module"""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_neighbors):
        super(PointnetFPModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        self.mlp = SharedMLP(in_channels, mlp_channels, ndim=1, bn=True)
        if num_neighbors == 0:
            self.interpolator = None
        elif num_neighbors == 3:
            self.interpolator = FeatureInterpolator(num_neighbors)
        else:
            raise ValueError('Expected value 1 or 3, but {} given.'.format(num_neighbors))

    def forward(self, dense_xyz, sparse_xyz, dense_feature, sparse_feature):
        if self.interpolator is None:
            assert sparse_xyz.size(2) == 1 and sparse_feature.size(2) == 1
            sparse_feature_expand = sparse_feature.expand(-1, -1, dense_xyz.size(2))
            new_feature = torch.cat([sparse_feature_expand, dense_feature], dim=1)
        else:
            new_feature = self.interpolator(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
        new_feature = self.mlp(new_feature)

        return new_feature

    def init_weights(self, init_fn=None):
        self.mlp.init_weights(init_fn)


class EdgeFPModule(nn.Module):
    """PointNet feature propagation module"""

    def __init__(self,
                 in_channels,
                 mlp_channels,
                 num_neighbors):
        super(EdgeFPModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        if num_neighbors == 0:
            self.interpolator = None
            self.mlp = SharedMLP(in_channels, mlp_channels, ndim=1, bn=True)

        elif num_neighbors == 3:
            self.interpolator = EdgeFeatureInterpolator(num_neighbors)
            self.mlp = SharedMLP(in_channels, mlp_channels, ndim=2, bn=True)

        else:
            raise ValueError('Expected value 1 or 3, but {} given.'.format(num_neighbors))

    def forward(self, dense_xyz, sparse_xyz, dense_feature, sparse_feature):
        if self.interpolator is None:
            assert sparse_xyz.size(2) == 1 and sparse_feature.size(2) == 1
            sparse_feature_expand = sparse_feature.expand(-1, -1, dense_xyz.size(2))
            new_feature = torch.cat([sparse_feature_expand, dense_feature], dim=1)
            new_feature = self.mlp(new_feature)
        else:
            new_feature = self.interpolator(dense_xyz, sparse_xyz, dense_feature, sparse_feature)
            new_feature = self.mlp(new_feature)
            new_feature = torch.mean(new_feature, dim=-1)

        return new_feature
