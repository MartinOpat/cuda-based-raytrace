#ifndef CUDA_ERROR_H
#define CUDA_ERROR_H

#include "cuda_runtime.h"

#define check_cuda_errors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t res, char const* const func, const char* const file, int const line);

#endif // CUDA_ERROR_H

