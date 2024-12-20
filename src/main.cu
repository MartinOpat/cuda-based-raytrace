#include "hurricanedata/datareader.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

// Not parallel computation
__global__ void computeMean(float *ans, size_t *masked_vals, size_t n, float *x) {
    float sum = 0;
    size_t num_not_masked_values = 0;
    size_t num_masked_values = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] < 1E14) { /* If x is not missing value */
            num_not_masked_values++;
            sum += x[i];
        } else {
            num_masked_values++;
        }
    }
    *ans = sum/num_not_masked_values;
    *masked_vals = num_masked_values;
}

int main() {
    std::string path = "data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4";
    std::string variable = "T";
    auto arr = loadDataToDevice(path, variable);

    float *ptr_mean;
    cudaMallocManaged(&ptr_mean, sizeof(float));

    size_t *ptr_masked;
    cudaMallocManaged(&ptr_masked, sizeof(size_t));

    computeMean<<<1, 1>>>(ptr_mean, ptr_masked, arr.second, arr.first);

    cudaDeviceSynchronize();

    std::cout << "Mean = " << *ptr_mean << " calculated from " << arr.second << " values where " << *ptr_masked << " are masked values.\n";

    cudaFree(arr.first);
    cudaFree(ptr_mean);
    cudaFree(ptr_masked);

    return 0;
}
