#ifndef _GROUP_POINTS
#define _GROUP_POINTS

#include <torch/extension.h>

// CUDA declarations
at::Tensor GroupPointsForward(
    const at::Tensor input,
    const at::Tensor index);

at::Tensor GroupPointsBackward(
    const at::Tensor grad_output,
    const at::Tensor index,
    const int64_t num_points);

#endif