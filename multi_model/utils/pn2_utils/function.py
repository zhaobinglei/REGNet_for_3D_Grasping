import torch
import pn2_ext
'''
try:
    from . import pn2_ext
    print("Import PointNet2 extension successfully")
except ImportError:
    print('Please compile source files before using pointnet2 cuda extension.')
'''

def gather_points(points, index):
    """Gather xyz of centroids according to indices

    Args:
        points: (batch_size, channels, num_points)
        index: (batch_size, num_centroids)

    Returns:
        new_xyz (torch.Tensor): (batch_size, channels, num_centroids)

    """
    batch_size = points.size(0)
    channels = points.size(1)
    num_centroids = index.size(1)
    index_expand = index.unsqueeze(1).expand(batch_size, channels, num_centroids)
    return points.gather(2, index_expand)


class FarthestPointSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_centroids):
        """Farthest point sample

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            num_centroids (int): the number of centroids to sample

        Returns:
            index (torch.Tensor): (batch_size, num_centroids), sample indices of centroids.

        """
        index = pn2_ext.farthest_point_sample(points, num_centroids)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None


farthest_point_sample = FarthestPointSample.apply


class BallQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, centroids, radius, num_neighbours):
        """Ball query

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            centroids (torch.Tensor): (batch_size, channels, num_centroids)
            radius (float): the radius of the ball
            num_neighbours (int): the number of neighbours within the ball.

        Returns:
            index (torch.Tensor): (batch_size, num_centroids, num_neighbours)
                indices of neighbours of each centroid.
            count (torch.Tensor): (batch_size, num_centroids)
                the number of unique neighbours of each centroid.

        """
        index, count = pn2_ext.ball_query(points, centroids, radius, num_neighbours)
        return index, count

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None


ball_query = BallQuery.apply


class GroupPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, index):
        """Group points by index

        Args:
            ctx:
            points (torch.Tensor): (batch_size, channels, num_points)
            index (torch.Tensor): (batch_size, num_centroids, num_neighbours), indices of neighbours of each centroid.

        Returns:
            group_points (torch.Tensor): (batch_size, channels, num_centroids, num_neighbours), grouped points.

        """
        ctx.save_for_backward(index)
        ctx.num_points = points.size(2)
        group_points = pn2_ext.group_points_forward(points, index)
        return group_points

    @staticmethod
    def backward(ctx, *grad_output):
        index = ctx.saved_tensors[0]
        grad_input = pn2_ext.group_points_backward(grad_output[0], index, ctx.num_points)
        return grad_input, None


group_points = GroupPoints.apply


class SearchNNDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query_xyz, key_xyz, num_neighbors):
        """For each point in query set, find its distances to k nearest neighbors in key set.

        Args:
            ctx:
            query_xyz: (B, 3, N1), xyz of the query points.
            key_xyz: (B, 3, N2), xyz of the key points.
            num_neighbors (int): k nearest neighbor

        Returns:
            index: (B, N1, K), indices of these neighbors in key_xyz.
            distance: (B, N1, K), distance to the k nearest neighbors in key_xyz.
        """
        index, distance = pn2_ext.point_search(query_xyz, key_xyz, num_neighbors)
        return index, distance

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None


search_nn_distance = SearchNNDistance.apply


# def search_nn_distance(query_xyz, key_xyz, num_neighbors):
#     from core.nn.functional import bpdist2
#     distance = bpdist2(query_xyz, key_xyz)
#     distance, index = torch.topk(distance, num_neighbors, dim=2, largest=False, sorted=True)
#     return index, distance


class FeatureInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, index, weight):
        """

        Args:
            ctx:
            feature: (B, C, N2), features of key points
            index: (B, N1, K), indices of key points to interpolate
            weight: (b, N1, K), weights to interpolate

        Returns:
            interpolated_feature: (B, C, N1)

        """
        _, _, num_inst = feature.size()
        ctx.save_for_backward(index, weight)
        ctx.num_inst = num_inst
        interpolated_feature = pn2_ext.interpolate_forward(feature, index, weight)
        return interpolated_feature

    @staticmethod
    def backward(ctx, *grad_out):
        index, weight = ctx.saved_tensors
        num_inst = ctx.num_inst
        grad_input = pn2_ext.interpolate_backward(grad_out[0], index, weight, num_inst)
        return grad_input, None, None


feature_interpolate = FeatureInterpolate.apply


# def feature_interpolate(feature, index, weight):
#     neighbour_feature = group_points(feature, index)
#     weighted_feature = neighbour_feature * weight.unsqueeze(1)
#     interpolated_feature = weighted_feature.sum(dim=-1)
#     return interpolated_feature
