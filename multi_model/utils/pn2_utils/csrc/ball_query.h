#ifndef _BALL_QUERY
#define _BALL_QUERY

#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> BallQuery(
    const at::Tensor points,
    const at::Tensor centroids,
    const float radius,
    const int64_t num_neighbours);

#endif