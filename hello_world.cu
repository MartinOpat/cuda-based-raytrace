#include <iostream>
#include <cuda_runtime.h>

#define cudaCheckError() {                                      \
    cudaError_t e = cudaGetLastError();                         \
    if (e != cudaSuccess) {                                     \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
               cudaGetErrorString(e));                          \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_from_gpu<<<1, 1>>>();
    cudaCheckError();

    cudaDeviceSynchronize();
    cudaCheckError();

    // Reset device
    cudaDeviceReset();
    return 0;
}
