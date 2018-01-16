#ifndef UTILS_H_
#define UTILS_H_

#include <iomanip>
#include <iostream>
#include <vector>

#include <cuda.h>

#define CUDA_CALL(f) { \
  ::cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout << #f ": " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  ::cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout << #f ": " << err << std::endl; \
    std::exit(1); \
  } \
}

template<typename T = void>
inline std::shared_ptr<T> allocate(std::size_t size) {
  T *ptr;
  CUDA_CALL(::cudaMalloc(&ptr, size));
  return std::shared_ptr<T>(ptr, [](T *ptr) { ::cudaFree(ptr); });
}

inline void print(const float *data, int n, int c, int h, int w) {
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

#endif  // UTILS_H_
