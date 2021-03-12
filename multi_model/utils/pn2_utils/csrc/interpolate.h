#ifndef _INTERPOLATE
#define _INTERPOLATE

#include <vector>
#include <torch/extension.h>

//CUDA declarations
std::vector<at::Tensor> PointSearch(
    const at::Tensor query_xyz,
    const at::Tensor key_xyz,
    const int64_t num_neighbours);

at::Tensor InterpolateForward(
    const at::Tensor input,
    const at::Tensor index,
    const at::Tensor weight);

at::Tensor InterpolateBackward(
    const at::Tensor grad_output,
    const at::Tensor index,
    const at::Tensor weight,
    const int64_t num_inst);

#endif
