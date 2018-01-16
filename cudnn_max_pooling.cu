#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

#include "utils.h"

const int x_w = 6;
const int x_h = 6;
const int x_c = 1;
const int x_n = 1;

const int win_w = 2;
const int win_h = 2;
const int pad_w = 0;
const int pad_h = 0;
const int str_w = 2;
const int str_h = 2;

const int x_bias = 1;

__global__ void dev_const(float *px, float k) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px, float bias) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid + bias;
}

int main() {
  ::cudnnHandle_t cudnn;
  CUDNN_CALL(::cudnnCreate(&cudnn));

  // input descriptor
  ::cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(::cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(::cudnnSetTensor4dDescriptor(
        x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w));

  // pooling descriptor
  ::cudnnPoolingDescriptor_t pool_desc;
  CUDNN_CALL(::cudnnCreatePoolingDescriptor(&pool_desc));
  CUDNN_CALL(::cudnnSetPooling2dDescriptor(
        pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
        win_h, win_w, pad_h, pad_w, str_h, str_w));

  // output sizes/descriptor
  int y_n, y_c, y_h, y_w;
  CUDNN_CALL(::cudnnGetPooling2dForwardOutputDim(
        pool_desc, x_desc, &y_n, &y_c, &y_h, &y_w));

  ::cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(::cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(::cudnnSetTensor4dDescriptor(
        y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w));

  // memories
  auto x_data = ::allocate<float>(x_n * x_c * x_h * x_w * sizeof(float));
  auto y_data = ::allocate<float>(y_n * y_c * y_h * y_w * sizeof(float));
  auto gy_data = ::allocate<float>(y_n * y_c * y_h * y_w * sizeof(float));
  auto gx_data = ::allocate<float>(x_n * x_c * x_h * x_w * sizeof(float));

  // initialize inputs
  dev_iota<<<x_w * x_h, x_n * x_c>>>(x_data.get(), x_bias);
  dev_const<<<y_w * y_h, y_n * y_c>>>(gy_data.get(), 1);
  dev_const<<<x_w * x_h, x_n * x_c>>>(gx_data.get(), 0);

  // perform forward operation
  float fwd_alpha = 1.f;
  float fwd_beta = 0.f;
  CUDNN_CALL(::cudnnPoolingForward(
        cudnn, pool_desc,
        &fwd_alpha, x_desc, x_data.get(),
        &fwd_beta, y_desc, y_data.get()));

  // perform backward operations
  float bwd_alpha = 1.f;
  float bwd_beta = 1.f;
  CUDNN_CALL(::cudnnPoolingBackward(
        cudnn, pool_desc,
        &bwd_alpha, y_desc, y_data.get(), y_desc, gy_data.get(),
        x_desc, x_data.get(), &bwd_beta, x_desc, gx_data.get()));

  // results
  std::cout << "x_w: " << x_w << std::endl;
  std::cout << "x_h: " << x_h << std::endl;
  std::cout << "x_c: " << x_c << std::endl;
  std::cout << "x_n: " << x_n << std::endl;
  std::cout << std::endl;
  std::cout << "win_w: " << win_w << std::endl;
  std::cout << "win_h: " << win_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << std::endl;
  std::cout << "y_w: " << y_w << std::endl;
  std::cout << "y_h: " << y_h << std::endl;
  std::cout << "y_c: " << y_c << std::endl;
  std::cout << "y_n: " << y_n << std::endl;
  std::cout << std::endl;

  std::cout << "x_data:" << std::endl;
  print(x_data.get(), x_n, x_c, x_h, x_w);
  std::cout << "y_data:" << std::endl;
  print(y_data.get(), y_n, y_c, y_h, y_w);
  std::cout << "gy_data:" << std::endl;
  print(gy_data.get(), y_n, y_c, y_h, y_w);
  std::cout << "gx_data:" << std::endl;
  print(gx_data.get(), x_n, x_c, x_h, x_w);

  // finalizing
  CUDNN_CALL(::cudnnDestroyTensorDescriptor(y_desc));
  CUDNN_CALL(::cudnnDestroyPoolingDescriptor(pool_desc));
  CUDNN_CALL(::cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(::cudnnDestroy(cudnn));
  return 0;
}
