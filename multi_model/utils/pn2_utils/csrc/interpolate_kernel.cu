// CUDA Implementation for feature interpolation

#ifndef _INTERPOLATING_KERNEL
#define _INTERPOLATING_KERNEL

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>

using at::cuda::detail::TensorInfo;
using at::cuda::detail::getTensorInfo;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define MAX_THREADS (uint64_t)512
#define K 3
#define TH_INDEX_BASE 0


/****************************
* Kernel for searching point
*****************************/
template <typename scalar_t, typename index_t>
__global__ void PointSearchKernel(
    const scalar_t *__restrict__ query_xyz,
    const scalar_t *__restrict__ key_xyz,
    index_t *__restrict__ index,
    scalar_t *__restrict__ distance,
    const int64_t num_query,  
    const int64_t num_key){

  const int batch_index = blockIdx.x;
  query_xyz += batch_index * num_query * 3;
  key_xyz += batch_index * num_key * 3;
  index += batch_index * num_query * K;
  distance += batch_index * num_query * K;

  for(int i = threadIdx.x; i < num_query; i += blockDim.x){ 
    int offset1 = i * 3;
    scalar_t x1 = query_xyz[offset1 + 0];
    scalar_t y1 = query_xyz[offset1 + 1];
    scalar_t z1 = query_xyz[offset1 + 2];

    scalar_t min_dist[K] = {1e40};
    int min_ind[K] = {-1};
    for (int j = 0; j < num_key; ++j) {
      int offset2 = j * 3;
      scalar_t x2 = key_xyz[offset2 + 0];
      scalar_t y2 = key_xyz[offset2 + 1];
      scalar_t z2 = key_xyz[offset2 + 2];
      scalar_t d = (x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2);
      // update min distance
      #pragma unroll
      for (int k = 0; k < K; ++k) {
        if (d < min_dist[k]) {
          for (int l = K - 1; l > k; --l) {
            min_dist[l] = min_dist[l - 1];
            min_ind[l] = min_ind[l - 1];
          }
          min_dist[k] = d;
          min_ind[k] = j;
          break;
        }
      }
    }
    #pragma unroll
    for (int k = 0; k < K; ++k) {
      index[offset1 + k] = min_ind[k];
      distance[offset1 + k] = min_dist[k];
    }
  }
}

/* PointSearch Interface
Input:
  query_xyz: (B, 3, N1)
  key_xyz: (B, 3, N2)
  k: number of neighbors
Output:
  index: (B, N1, K)
  distance: (B, N1, K)
*/
std::vector<at::Tensor> PointSearch(
    const at::Tensor query_xyz,
    const at::Tensor key_xyz,
    const int64_t num_neighbours) {
  const auto batch_size = query_xyz.size(0);
  const auto num_query = query_xyz.size(2);
  const auto num_key = key_xyz.size(2);

  // sanity check
  CHECK_EQ(key_xyz.size(0), batch_size);
  CHECK_EQ(query_xyz.size(1), 3);
  CHECK_EQ(key_xyz.size(1), 3);
  // Only support 3-nn
  CHECK_EQ(num_neighbours, K);
  CHECK_GE(num_key, num_neighbours);
  CHECK_CUDA(query_xyz);
  CHECK_CUDA(key_xyz);

  // Convert the Tensor Dimension
  auto query_xyz_trans = query_xyz.transpose(1, 2).contiguous();  // (B, N1, 3)
  auto key_xyz_trans = key_xyz.transpose(1, 2).contiguous(); // (B, N2, 3)
  auto index = at::zeros({batch_size, num_query, num_neighbours}, query_xyz.type().toScalarType(at::kLong));
  auto distance = at::zeros({batch_size, num_query, num_neighbours}, query_xyz.type());
  
  const auto block = std::min((uint64_t)num_key, MAX_THREADS);

  AT_DISPATCH_FLOATING_TYPES(query_xyz.type(), "PointSearch", ([&] {
    PointSearchKernel<scalar_t, int64_t>
      <<<batch_size, block>>>(
        query_xyz_trans.data<scalar_t>(),
        key_xyz_trans.data<scalar_t>(),
        index.data<int64_t>(),
        distance.data<scalar_t>(),
        num_query,
        num_key);
    }));
  
  THCudaCheck(cudaGetLastError());
  
  return std::vector<at::Tensor>({index, distance});
}


/********************************
* Forward kernel for interpolate
*********************************/
template<typename scalar_t, typename index_t>
__global__ void InterpolateForwardKernel(
    const TensorInfo<scalar_t, uint64_t> output,
    const TensorInfo<scalar_t, uint64_t> input,
    const TensorInfo<index_t, uint64_t> index,
    const TensorInfo<scalar_t, uint64_t> weight,
    const uint64_t totalElements){
   
  uint64_t channels = input.sizes[1];
  uint64_t num_inst = input.sizes[2];
  uint64_t num_select = index.sizes[1];
  // uint64_t k = index.sizes[2];
  for (uint64_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
      linearId < totalElements;
      linearId += gridDim.x * blockDim.x) {
    // Compute offsets
    uint64_t linearId_tmp = linearId;
    uint64_t inst_offset = linearId_tmp % num_select;
    linearId_tmp /= num_select;
    uint64_t channel_offset = linearId_tmp % channels;
    uint64_t batch_offset = linearId_tmp / channels;
    
    scalar_t outputValue = 0.0;
    uint64_t srcOffset = channel_offset * input.strides[1]
      + batch_offset * input.strides[0];
    uint64_t indexOffset = inst_offset * index.strides[1]
      + batch_offset * index.strides[0];
    uint64_t weightOffset = inst_offset * weight.strides[1]
      + batch_offset * weight.strides[0];

    #pragma unroll
    for (int k = 0; k < K; ++k) {
      index_t indexValue = index.data[indexOffset + k * index.strides[2]] - TH_INDEX_BASE;
      scalar_t weightValue = weight.data[weightOffset + k * weight.strides[2]];
      assert(indexValue >= 0 && indexValue < num_inst);
      outputValue += input.data[srcOffset + indexValue * input.strides[2]] * weightValue;
    }

    uint64_t tensorOffset = inst_offset * output.strides[2] 
      + channel_offset * output.strides[1]
      + batch_offset * output.strides[0];
    output.data[tensorOffset] = outputValue;
  }
}

/* Interpolate forward interface
Input:
  input: (B, C, M)
  index: (B, N, K), k is the number of neighbors in PointSearch
  weight: (B, N, K)
Output:
  output: (B, C, N)
*/
at::Tensor InterpolateForward(
    const at::Tensor input,
    const at::Tensor index,
    const at::Tensor weight){
  const auto batch_size = input.size(0);
  const auto channels = input.size(1);
  const auto num_inst = input.size(2);
  const auto num_select = index.size(1);

  auto output = at::zeros({batch_size, channels, num_select}, input.type());
  CHECK_EQ(index.size(0), batch_size);
  CHECK_EQ(index.size(2), K);
  CHECK_EQ(weight.size(0), batch_size);
  CHECK_EQ(weight.size(1), num_select);
  CHECK_EQ(weight.size(2), K);
  CHECK_CUDA(input);
  CHECK_CUDA(index);
  CHECK_CUDA(weight);
  CHECK_CONTIGUOUS(output);

  // Calculate grids and blocks for kernels 
  const auto totalElements = output.numel();
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;
  const int curDevice = at::cuda::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(totalElements, grid, curDevice), 1, "Too many elements to calculate");

  AT_DISPATCH_FLOATING_TYPES(input.type(), "InterpolateForward", ([&] {
    auto ouputInfo = getTensorInfo<scalar_t, uint64_t>(output);
    auto inputInfo = getTensorInfo<scalar_t, uint64_t>(input);
    auto indexInfo = getTensorInfo<int64_t, uint64_t>(index);
    auto weightInfo = getTensorInfo<scalar_t, uint64_t>(weight);
    InterpolateForwardKernel<scalar_t, int64_t>
      <<<grid, block>>>(
        ouputInfo,
        inputInfo,
        indexInfo,
        weightInfo,
        (uint64_t)totalElements);
    }));
  
  THCudaCheck(cudaGetLastError());

  return output;
}  


/**********************************
* Backward kernel for interpolate 
***********************************/
/* Backward Kernel */
template <typename scalar_t, typename index_t>
__global__ void InterpolateBackwardKernel(
    const TensorInfo<scalar_t, uint64_t> grad_input,
    const TensorInfo<scalar_t, uint64_t> grad_output,
    const TensorInfo<index_t, uint64_t> index,
    const TensorInfo<scalar_t, uint64_t> weight,
    const index_t totalElements) {
  uint64_t channels = grad_input.sizes[1];
  uint64_t num_inst = grad_input.sizes[2];
  uint64_t num_select = index.sizes[1];
  // index_t k = index.sizes[2];
  for (index_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    // Compute offsets
    uint64_t linearId_tmp = linearId;
    uint64_t inst_offset = linearId_tmp % num_select;
    linearId_tmp /= num_select;
    uint64_t channel_offset = linearId_tmp % channels;
    uint64_t batch_offset = linearId_tmp / channels;
    
    uint64_t srcOffset = inst_offset * grad_output.strides[2]
      + channel_offset * grad_output.strides[1]
      + batch_offset * grad_output.strides[0];

    uint64_t tensorOffset = channel_offset * grad_input.strides[1]
      + batch_offset * grad_input.strides[0];
    
    uint64_t indexOffset = inst_offset * index.strides[1]
      + batch_offset * index.strides[0];

    uint64_t weightOffset = inst_offset * weight.strides[1]
      + batch_offset * weight.strides[0];

    scalar_t gradValue = grad_output.data[srcOffset];
    #pragma unroll
    for (int k = 0; k < K; ++k) {
      index_t indexValue = index.data[indexOffset + k * index.strides[2]] - TH_INDEX_BASE;
      scalar_t weightValue = weight.data[weightOffset + k * weight.strides[2]];
      assert(indexValue >= 0 && indexValue < num_inst);
      atomicAdd(&grad_input.data[tensorOffset + indexValue * grad_input.strides[2]], gradValue * weightValue);
    }
  }
}

/* Interpolate backward interface
Input:
  grad_output: (B, C, M)
  index: (B, M, K)
  weight: (B, M, K)
Output:
  grad_input: (B, C, N)
*/
at::Tensor InterpolateBackward(
    const at::Tensor grad_output,
    const at::Tensor index,
    const at::Tensor weight,
    const int64_t num_inst){
  const auto batch_size = grad_output.size(0);
  const auto channels = grad_output.size(1);
  const auto num_select = grad_output.size(2);

  auto grad_input = at::zeros({batch_size, channels, num_inst}, grad_output.type());
  CHECK_EQ(index.size(0), batch_size);
  CHECK_EQ(index.size(2), K);
  CHECK_EQ(weight.size(0), batch_size);
  CHECK_EQ(weight.size(1), num_select);
  CHECK_EQ(weight.size(2), K);
  CHECK_CUDA(grad_output);
  CHECK_CUDA(index);
  CHECK_CUDA(weight);
  CHECK_CONTIGUOUS(grad_input);

  // Calculate grids and blocks for kernels 
  const auto totalElements = grad_output.numel();
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;
  const int curDevice = at::cuda::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(totalElements, grid, curDevice), 1, "Too many elements to calculate");

  AT_DISPATCH_FLOATING_TYPES(grad_output.type(), "InterpolateBackward", ([&] {
    auto gradInputInfo = getTensorInfo<scalar_t, uint64_t>(grad_input);
    auto gradOutputInfo = getTensorInfo<scalar_t, uint64_t>(grad_output);
    auto IndexInfo = getTensorInfo<int64_t, uint64_t>(index);
    auto weightInfo = getTensorInfo<scalar_t, uint64_t>(weight);
    InterpolateBackwardKernel<scalar_t, int64_t>
      <<<grid, block>>>(
        gradInputInfo,
        gradOutputInfo,
        IndexInfo,
        weightInfo,
        (uint64_t)totalElements);
  }));

  THCudaCheck(cudaGetLastError());

  return grad_input;
}

#endif
