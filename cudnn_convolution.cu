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
  CUDNN_CALL(cudnnCreate(&cudnn));

  // input
  cudnnTensorDescriptor_t x_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w));

  // filter
  cudnnFilterDescriptor_t w_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&w_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w_k, w_c, w_h, w_w));

  // convolution
  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
#if CUDNN_MAJOR >= 6
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));
#else
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION));
#endif  // CUDNN_MAJOR

  // output
  int y_n, y_c, y_h, y_w;
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, x_desc, w_desc, &y_n, &y_c, &y_h, &y_w));

  cudnnTensorDescriptor_t y_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        x_desc, w_desc, conv_desc, y_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size));

  // perform
  float alpha = 1.f;
  float beta = 0.f;

  auto x_data = ::allocate<float>(x_n * x_c * x_h * x_w * sizeof(float));
  auto w_data = ::allocate<float>(w_k * w_c * w_h * w_w * sizeof(float));
  auto y_data = ::allocate<float>(y_n * y_c * y_h * y_w * sizeof(float));
  auto ws_data = ::allocate(ws_size);

  dev_iota<<<x_w * x_h, x_n * x_c>>>(x_data.get(), x_bias);
  dev_iota<<<w_w * w_h, w_k * w_c>>>(w_data.get(), w_bias);
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, x_desc, x_data.get(), w_desc, w_data.get(),
      conv_desc, algo, ws_data.get(), ws_size,
      &beta, y_desc, y_data.get()));

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

  std::cout << "Convolution algorithm: " << algo << std::endl;
  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  std::cout << "x_data:" << std::endl;
  print(x_data.get(), x_n, x_c, x_h, x_w);

  std::cout << "w_data:" << std::endl;
  print(w_data.get(), w_k, w_c, w_h, w_w);

  std::cout << "y_data:" << std::endl;
  print(y_data.get(), y_n, y_c, y_h, y_w);

  // finalizing
  CUDNN_CALL(cudnnDestroyTensorDescriptor(y_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(w_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(x_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}
