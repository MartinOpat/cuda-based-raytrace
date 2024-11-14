#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    hello_from_gpu<<<1, 1>>>();

    cudaDeviceSynchronize();

    // Reset device
    cudaDeviceReset();
    return 0;
}
