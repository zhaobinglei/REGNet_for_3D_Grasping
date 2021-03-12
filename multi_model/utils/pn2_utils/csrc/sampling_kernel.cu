/* CUDA Implementation for sampling*/
#ifndef _SAMPLING_KERNEL
#define _SAMPLING_KERNEL

#include <cmath>

#include <ATen/ATen.h>
#include <THC/THC.h>

// NOTE: AT_ASSERT has become AT_CHEAK on master after 0.4.
// NOTE: AT_CHEAK has become TORCH_CHECK on master after 1.x.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_EQ(x, y) TORCH_CHECK(x == y, #x " does not equal to " #y)
// #define CHECK_GT(x, y) TORCH_CHECK(x > y, #x " is not greater than " #y)

#define CASE_RUN(BLOCK) \
  case BLOCK: \
    AT_DISPATCH_FLOATING_TYPES(points.type(), "FarthestPointSample", ([&] { \
      FarthestPointSampleKernel<BLOCK, scalar_t, int64_t> \
        <<<batch_size, BLOCK>>>( \
        index.data<int64_t>(), \
        points_trans.data<scalar_t>(), \
        temp.data<scalar_t>(), \
        num_points, \
        num_centroids); \
    })); \
    break;

#define MAX_THREADS uint64_t(512)

inline uint64_t get_block(int64_t x) {
  int cnt = 0;
  x -= 1;
  while (x > 0) {
    x = x >> 1;
    cnt += 1;
  }
  return std::min(uint64_t(1) << cnt, MAX_THREADS);
}

/*
points: (B, N1, 3)
temp: (B, N1)
index: (B, N2)
*/
template <unsigned int block_size, typename scalar_t, typename index_t>
__global__ void FarthestPointSampleKernel(
    index_t* __restrict__ index,
    const scalar_t* __restrict__ points,
    scalar_t* __restrict__ temp,
    const int64_t num_points,
    const int64_t num_centroids) {
  // alocated shared memory
  __shared__ scalar_t smem_dist[block_size];
  // use int32 to save memory
  __shared__ int32_t smem_ind[block_size];

  const int batch_index = blockIdx.x;
  int32_t cur_ind = 0;
  const scalar_t* points_offset = points + batch_index * num_points * 3;
  scalar_t* temp_offset = temp + batch_index * num_points;
  index_t* index_offset = index + batch_index * num_centroids;
  // explicitly choose the first point as a centroid
  if (threadIdx.x == 0) index_offset[0] = cur_ind;
  
  for (int i = 1; i < num_centroids; ++i) {
    scalar_t max_dist = 0;
    int32_t max_ind = cur_ind;
    
    int32_t offset1 = cur_ind * 3;
    scalar_t x1 = points_offset[offset1 + 0];
    scalar_t y1 = points_offset[offset1 + 1];
    scalar_t z1 = points_offset[offset1 + 2];

    for (int j = threadIdx.x; j < num_points; j += block_size) {
      int32_t offset2 = j * 3;
      scalar_t x2 = points_offset[offset2 + 0];
      scalar_t y2 = points_offset[offset2 + 1];
      scalar_t z2 = points_offset[offset2 + 2];

      scalar_t dist = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
      scalar_t last_dist = temp_offset[j];
      if (last_dist > dist || last_dist < 0) {
        temp_offset[j] = dist;
      } else {
        dist = last_dist;
      }
      if (dist > max_dist) {
        max_dist = dist;
        max_ind = j;
      }
    }

    smem_dist[threadIdx.x] = max_dist;
    smem_ind[threadIdx.x] = max_ind;

    // assert block_size == blockDim.x
    int offset = block_size / 2;
    while (offset > 0) {
      __syncthreads();
      if (threadIdx.x < offset) {
        scalar_t dist1 =  smem_dist[threadIdx.x];
        scalar_t dist2 = smem_dist[threadIdx.x+offset];
        if (dist1 < dist2) {
          smem_dist[threadIdx.x] = dist2;
          smem_ind[threadIdx.x] = smem_ind[threadIdx.x+offset];
        }
      }
      offset /= 2;
    }
    __syncthreads();

    cur_ind = smem_ind[0];
    if (threadIdx.x == 0) index_offset[i] = (index_t)cur_ind;
  }
}

/*
Only forward is required.
Input:
  points: (B, 3, N1)
Output:
  index: (B, N2)
*/
at::Tensor FarthestPointSample(
	const at::Tensor points,
    const int64_t num_centroids) {

	const auto batch_size = points.size(0);
	const auto num_points = points.size(2);

	// Sanity check
	CHECK_CUDA(points);
	CHECK_EQ(points.size(1), 3);
  CHECK_GT(num_centroids, 0);
  CHECK_GE(num_points, num_centroids);
	
  auto points_trans = points.transpose(1, 2).contiguous();  // (B, N1, 3)
  auto index = at::zeros({batch_size, num_centroids}, points.type().toScalarType(at::kLong));
  // In original implementation, it only allocates memory with the size of grid instead of batch size.
  auto temp = at::neg(at::ones({batch_size, num_points}, points.type()));

  // In order to make full use of shared memory and threads,
  // it is recommended to set num_centroids to be power of 2.
  const auto block = get_block(num_points);

  switch (block) {
    CASE_RUN(512)
    CASE_RUN(256)
    CASE_RUN(128)
    CASE_RUN(64)
    CASE_RUN(32)
    CASE_RUN(16)
    default:
      AT_DISPATCH_FLOATING_TYPES(points.type(), "FarthestPointSample", ([&] {
      FarthestPointSampleKernel<16, scalar_t, int64_t>
        <<<batch_size, 16>>>(
        index.data<int64_t>(),
        points_trans.data<scalar_t>(),
        temp.data<scalar_t>(),
        num_points,
        num_centroids);
    }));
  }

  THCudaCheck(cudaGetLastError());
  
  return index;
}

#endif