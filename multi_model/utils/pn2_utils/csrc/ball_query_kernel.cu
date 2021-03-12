/* CUDA Implementation for ball query*/
#ifndef _BALL_QUERY_KERNEL
#define _BALL_QUERY_KERNEL

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <THC/THC.h>

// NOTE: AT_ASSERT has become AT_CHEAK on master after 0.4.
// NOTE: AT_CHEAK has become TORCH_CHECK on master after 1.x.
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// #define CHECK_EQ(x, y) TORCH_CHECK(x == y, #x " does not equal to " #y)
// #define CHECK_GT(x, y) TORCH_CHECK(x > y, #x " is not greater than " #y)

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

template <typename scalar_t, typename index_t>
__global__ void BallQueryKernel(
    index_t* __restrict__ index,
    index_t* __restrict__ count,
    const scalar_t *__restrict__ points,
    const int64_t num_points,
    const scalar_t *__restrict__ centroids,
    const int64_t num_centroids,
    const scalar_t radius,
    const int64_t num_neighbours) {
  const int batch_index = blockIdx.x;
  index += batch_index * num_centroids * num_neighbours;
  count += batch_index * num_centroids;
  points += batch_index * num_points * 3;
  centroids += batch_index * num_centroids * 3;
  
  scalar_t radius_square = radius * radius;
  for (int i = threadIdx.x; i < num_centroids; i += blockDim.x) {
    int offset1 = i * 3;
    int offset3 = i * num_neighbours;
    scalar_t x1 = centroids[offset1 + 0];
    scalar_t y1 = centroids[offset1 + 1];
    scalar_t z1 = centroids[offset1 + 2];
    index_t cnt = 0;
    for (int j = 0; j < num_points && cnt < num_neighbours; ++j) {
      int offset2 = j * 3;
      scalar_t x2 = points[offset2 + 0];
      scalar_t y2 = points[offset2 + 1];
      scalar_t z2 = points[offset2 + 2];
      scalar_t dist = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
      if (dist < radius_square) {
        if (cnt == 0) {
          for (int k = 0; k < num_neighbours; ++k) {
            index[offset3 + k] = j;
          }
        } else {
          index[offset3 + cnt] = j;
        }
        ++cnt;
      }
    }
    count[i] = cnt;
  }
}

/*
Only forward is required.
Input:
  points: (B, 3, N1)
  centroids: (B, 3, N2)
  raidus: scalar
  num_neighbours: int
Output:
  index: (B, N2, N3)
  count: (B, N2)
*/
std::vector<at::Tensor> BallQuery(
    const at::Tensor points,
    const at::Tensor centroids,
    const float radius,
    const int64_t num_neighbours) {

  const auto batch_size = points.size(0);
  const auto num_points = points.size(2);
  const auto num_centroids = centroids.size(2);

  // Sanity check
  CHECK_CUDA(points);
  CHECK_CUDA(centroids);
  CHECK_EQ(points.size(1), 3);
  CHECK_EQ(centroids.size(1), 3);
  
  auto points_trans = points.transpose(1, 2).contiguous();  // (B, N1, 3)
  auto centroids_trans = centroids.transpose(1, 2).contiguous();  // (B, N2, 3)

  // Allocate new space for output
  auto index = at::zeros({batch_size, num_centroids, num_neighbours}, points.type().toScalarType(at::kLong));
  index.set_requires_grad(false);
  auto count = at::zeros({batch_size, num_centroids}, index.type());
  CHECK_CUDA(index); CHECK_CONTIGUOUS(index);
  CHECK_CUDA(count); CHECK_CONTIGUOUS(count);

  const auto block = get_block(num_centroids);

  AT_DISPATCH_FLOATING_TYPES(points.type(), "BallQuery", ([&] {
    BallQueryKernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
      index.data<int64_t>(),
      count.data<int64_t>(),
      points_trans.data<scalar_t>(),
      num_points,
      centroids_trans.data<scalar_t>(),
      num_centroids,
      (scalar_t)radius,
      num_neighbours);
  }));

  THCudaCheck(cudaGetLastError());

  return std::vector<at::Tensor>({index, count});
}

#endif