#include "cuda_error.h"

#include "cuda_runtime.h"
#include <iostream>


void check_cuda(cudaError_t res, char const* const func, const char* const file, int const line) {
  if (res) {
    std::cout << "CUDA encountered an error: " << cudaGetErrorString(res) << " in " << file << ":" << line << std::endl;
    cudaDeviceReset();
    exit(1);
  }
}
