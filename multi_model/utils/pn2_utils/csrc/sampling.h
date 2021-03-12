#ifndef _SAMPLING
#define _SAMPLING

#include <torch/extension.h>

// CUDA declarations
at::Tensor FarthestPointSample(
    const at::Tensor points,
    const int64_t num_centroids);

#endif