#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

const int x_n = 1;
const int x_c = 1;
const int x_h = 5;
const int x_w = 5;

const int w_k = 1;
const int w_c = 1;
const int w_h = 2;
const int w_w = 2;

const int pad_h = 0;
const int pad_w = 0;
const int str_h = 1;
const int str_w = 1;
const int dil_h = 1;
const int dil_w = 1;

const int x_bias = 1;
const int w_bias = 1;

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << #f ": " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << #f ": " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px, float bias) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid + bias;
}

template<typename T = void>
std::shared_ptr<T> allocate(std::size_t size) {
  T *ptr;
  CUDA_CALL(::cudaMalloc(&ptr, size));
  return std::shared_ptr<T>(ptr, [](T *ptr) { ::cudaFree(ptr); });
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(::cudaMemcpy(
        buffer.data(), data, n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(6) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main() {
  cudnnHandle_t cudnn;
  CUDNN_CALL(::cudnnCreate(&cudnn));

  // input
  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(::cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(::cudnnSetTensor4dDescriptor(
        x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w));

  // filter
  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(::cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(::cudnnSetFilter4dDescriptor(
        w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w));

  // convolution
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(::cudnnCreateConvolutionDescriptor(&conv_desc));
#if CUDNN_MAJOR >= 6
  CUDNN_CALL(::cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
#else
  CUDNN_CALL(::cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION));
#endif  // CUDNN_MAJOR

  // output
  int y_n, y_c, y_h, y_w;
  CUDNN_CALL(::cudnnGetConvolution2dForwardOutputDim(
        conv_desc, x_desc, w_desc, &y_n, &y_c, &y_h, &y_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(::cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(::cudnnSetTensor4dDescriptor(
        y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w));

  // algorithms
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_x_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_w_algo;
  CUDNN_CALL(::cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        x_desc, w_desc, conv_desc, y_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo));
  CUDNN_CALL(::cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn,
        w_desc, y_desc, conv_desc, x_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_x_algo));
  CUDNN_CALL(::cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn,
        x_desc, y_desc, conv_desc, w_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwd_w_algo));

  // workspaces
  size_t fwd_ws_size, bwd_x_ws_size, bwd_w_ws_size;
  CUDNN_CALL(::cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, x_desc, w_desc, conv_desc, y_desc, fwd_algo, &fwd_ws_size));
  CUDNN_CALL(::cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn, w_desc, y_desc, conv_desc, x_desc, bwd_x_algo, &bwd_x_ws_size));
  CUDNN_CALL(::cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn, x_desc, y_desc, conv_desc, w_desc, bwd_w_algo, &bwd_w_ws_size));

  // memories
  auto x_data = ::allocate<float>(x_n * x_c * x_h * x_w * sizeof(float));
  auto w_data = ::allocate<float>(w_k * w_c * w_h * w_w * sizeof(float));
  auto y_data = ::allocate<float>(y_n * y_c * y_h * y_w * sizeof(float));
  auto gy_data = ::allocate<float>(y_n * y_c * y_h * y_w * sizeof(float));
  auto gx_data = ::allocate<float>(x_n * x_c * x_h * x_w * sizeof(float));
  auto gw_data = ::allocate<float>(w_k * w_c * w_h * w_w * sizeof(float));
  auto fwd_ws_data = ::allocate(fwd_ws_size);
  auto bwd_x_ws_data = ::allocate(bwd_x_ws_size);
  auto bwd_w_ws_data = ::allocate(bwd_w_ws_size);

  // initialize inputs
  dev_iota<<<x_w * x_h, x_n * x_c>>>(x_data.get(), x_bias);
  dev_iota<<<w_w * w_h, w_k * w_c>>>(w_data.get(), w_bias);
  dev_const<<<y_w * y_h, y_n * y_c>>>(gy_data.get(), 1);
  dev_const<<<x_w * x_h, x_n * x_c>>>(gx_data.get(), 0);
  dev_const<<<w_w * w_h, w_k * w_c>>>(gw_data.get(), 0);

  // perform forward operation
  float fwd_alpha = 1.f;
  float fwd_beta = 0.f;
  CUDNN_CALL(::cudnnConvolutionForward(
        cudnn,
        &fwd_alpha, x_desc, x_data.get(), w_desc, w_data.get(),
        conv_desc, fwd_algo, fwd_ws_data.get(), fwd_ws_size,
        &fwd_beta, y_desc, y_data.get()));

  // perform backward operations
  float bwd_alpha = 1.f;
  float bwd_beta = 1.f;
  CUDNN_CALL(::cudnnConvolutionBackwardData(
        cudnn,
        &bwd_alpha, w_desc, w_data.get(), y_desc, gy_data.get(),
        conv_desc, bwd_x_algo, bwd_x_ws_data.get(), bwd_x_ws_size,
        &bwd_beta, x_desc, gx_data.get()));
  CUDNN_CALL(::cudnnConvolutionBackwardFilter(
        cudnn,
        &bwd_alpha, x_desc, x_data.get(), y_desc, gy_data.get(),
        conv_desc, bwd_w_algo, bwd_w_ws_data.get(), bwd_w_ws_size,
        &bwd_beta, w_desc, gw_data.get()));

  // results
  std::cout << "x_n: " << x_n << std::endl;
  std::cout << "x_c: " << x_c << std::endl;
  std::cout << "x_h: " << x_h << std::endl;
  std::cout << "x_w: " << x_w << std::endl;
  std::cout << std::endl;
  std::cout << "w_k: " << w_k << std::endl;
  std::cout << "w_c: " << w_c << std::endl;
  std::cout << "w_h: " << w_h << std::endl;
  std::cout << "w_w: " << w_w << std::endl;
  std::cout << std::endl;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;
  std::cout << "y_n: " << y_n << std::endl;
  std::cout << "y_c: " << y_c << std::endl;
  std::cout << "y_h: " << y_h << std::endl;
  std::cout << "y_w: " << y_w << std::endl;
  std::cout << std::endl;

  std::cout << "Algorithm (fwd): " << fwd_algo << std::endl;
  std::cout << "Algorithm (bwd-x): " << bwd_x_algo << std::endl;
  std::cout << "Algorithm (bwd-w): " << bwd_w_algo << std::endl;
  std::cout << "Workspace size (fwd): " << fwd_ws_size << std::endl;
  std::cout << "Workspace size (bwd-x): " << bwd_x_ws_size << std::endl;
  std::cout << "Workspace size (bwd-w): " << bwd_w_ws_size << std::endl;
  std::cout << std::endl;

  std::cout << "x_data:" << std::endl;
  print(x_data.get(), x_n, x_c, x_h, x_w);
  std::cout << "w_data:" << std::endl;
  print(w_data.get(), w_k, w_c, w_h, w_w);
  std::cout << "y_data:" << std::endl;
  print(y_data.get(), y_n, y_c, y_h, y_w);
  std::cout << "gy_data:" << std::endl;
  print(gy_data.get(), y_n, y_c, y_h, y_w);
  std::cout << "gx_data:" << std::endl;
  print(gx_data.get(), x_n, x_c, x_h, x_w);
  std::cout << "gw_data:" << std::endl;
  print(gw_data.get(), w_k, w_c, w_h, w_w);

  // finalizing
  CUDNN_CALL(::cudnnDestroyTensorDescriptor(y_desc));
  CUDNN_CALL(::cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(::cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(::cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(::cudnnDestroy(cudnn));
  return 0;
}
