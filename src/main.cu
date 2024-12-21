#include "hurricanedata/fielddata.h"
#include "hurricanedata/gpubuffer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <iomanip> 


// Not parallel computation
__global__ void computeMean(float *ans, const FieldMetadata &fmd, FieldData fd) {
    float sum = 0;
    size_t num_not_masked_values = 0;
    for (int i = 0; i < fmd.widthSize; i++) {
        double xi = getVal(fmd, fd, 2, 20, 100, i);
        if (xi < 1E14) { /* If x is not missing value */
            num_not_masked_values++;
            sum += xi;
        }
    }
    *ans = sum/num_not_masked_values;
}

int main() {
    std::string path = "data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4";
    std::string variable = "T";
    GPUBuffer buffer{path, variable};

    auto fd = buffer.nextFieldData();

    float *ptr_mean;
    cudaMallocManaged(&ptr_mean, sizeof(float));

    computeMean<<<1, 1>>>(ptr_mean, *buffer.fmd, fd);

    cudaDeviceSynchronize();

    std::cout << "Mean = " << std::fixed << std::setprecision(6) << *ptr_mean << "\n";

    cudaFree(fd.valArrays[0]);
    cudaFree(ptr_mean);
    return 0;
}
