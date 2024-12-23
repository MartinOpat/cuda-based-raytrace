// #include "hurricanedata/fielddata.h"
// #include "hurricanedata/gpubufferhandler.h"
#include "hurricanedata/datareader.h"
#include "hurricanedata/gpubuffer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <iomanip> 


// Not parallel computation
// __global__ void computeMean(float *ans, const FieldMetadata &fmd, FieldData fd) {
//     float sum = 0;
//     size_t num_not_masked_values = 0;
//     for (int i = 0; i < fmd.widthSize; i++) {
//         double xi = getVal(fmd, fd, 2, 20, 100, i);
//         if (xi < 1E14) { /* If x is not missing value */
//             num_not_masked_values++;
//             sum += xi;
//         }
//     }
//     *ans = sum/num_not_masked_values;
// }

__global__ void computeMean(float *ans, DataHandle dh) {
    float sum = 0;
    size_t num_not_masked_values = 0;
    for (int i = 0; i < dh.size; i++) {
        double xi = dh.d_data[i];
        if (xi < 1E14) { /* If x is not missing value */
            num_not_masked_values++;
            sum += xi;
        }
    }
    *ans = sum/num_not_masked_values;
}

int main() {
    std::string path = "data";

    std::string variable = "T";

    // std::unique_ptr<DataReader> dataReader = std::make_unique<DataReader>(path, variable);
    DataReader dataReader{path, variable};

    std::cout << "created datareader\n";

    GPUBuffer buffer (dataReader);

    std::cout << "created buffer\n";

    auto dataHandle = buffer.getDataHandle(0);

    std::cout << "got a data handle\n";

    auto x = buffer.getAxis<int>(0, "time");
    std::cout << "size of x=" << x.first << "\n";
    std::cout << "x[1]= " <<x.second[1] << "\n";


    // GPUBufferHandler buffer{path, variable};

    // auto fd = buffer.nextFieldData();

    float *ptr_mean;
    cudaMallocManaged(&ptr_mean, sizeof(float));

    computeMean<<<1, 1>>>(ptr_mean, dataHandle);

    cudaDeviceSynchronize();

    std::cout << "Mean = " << std::fixed << std::setprecision(6) << *ptr_mean << "\n";

    // cudaFree(fd.valArrays[0]);
    cudaFree(ptr_mean);
    return 0;
}
