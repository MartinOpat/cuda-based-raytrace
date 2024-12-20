#include "hurricanedata/gpubuffer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

__device__ float getVal(
    const FieldMetadata &md,
    const FieldData &d,
    const size_t &timeInd,
    const size_t &lonInd,
    const size_t &latInd,
    const size_t &levInd
) {
    // TODO: Actaully implement function
    return d.valArrays[0][timeInd]; 
}

// Not parallel computation
__global__ void computeMean(float *ans, size_t *masked_vals, const FieldMetadata &fmd, FieldData fd) {
    float sum = 0;
    size_t num_not_masked_values = 0;
    size_t num_masked_values = 0;
    for (int i = 0; i < fmd.widthSize*fmd.heightSize*fmd.depthSize*fd.timeSize; i++) {
        double xi = getVal(fmd, fd, i, 0, 0, 0);
        if (xi < 1E14) { /* If x is not missing value */
            num_not_masked_values++;
            sum += xi;
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
    GPUBuffer buffer{path, variable};

    auto fd = buffer.nextFieldData();

    float *ptr_mean;
    cudaMallocManaged(&ptr_mean, sizeof(float));

    size_t *ptr_masked;
    cudaMallocManaged(&ptr_masked, sizeof(size_t));

    computeMean<<<1, 1>>>(ptr_mean, ptr_masked, *buffer.fmd, fd);

    cudaDeviceSynchronize();

    std::cout << "Mean = " << *ptr_mean << " values where " << *ptr_masked << " are masked values.\n";

    cudaFree(fd.valArrays[0]);
    cudaFree(ptr_mean);
    cudaFree(ptr_masked);
    return 0;
}
