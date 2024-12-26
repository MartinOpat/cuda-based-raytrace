// #include "hurricanedata/fielddata.h"
#include "hurricanedata/gpubufferhandler.h"
#include "hurricanedata/datareader.h"
#include "hurricanedata/gpubuffer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <iomanip> 

__global__ void getSingleValue(float *ans, const FieldMetadata &fmd, FieldData fd) {
    float xi = getVal(fmd, fd, 1, 20, 100, 100);
    *ans = xi;
}

int main() {
    std::string path = "data/atmosphere_MERRA-wind-speed[179253532]";

    std::string variable = "T";

    // std::unique_ptr<DataReader> dataReader = std::make_unique<DataReader>(path, variable);
    DataReader dataReader{path, variable};

    std::cout << "created datareader\n";

    GPUBuffer buffer (dataReader);

    std::cout << "created buffer\n";

    GPUBufferHandler bufferHandler(buffer);

    float *ptr_test_read;
    cudaMallocManaged(&ptr_test_read, sizeof(float));

    std::cout << "created buffer handler\n";
    for (int i = 0; i < 10; i++) {
        FieldData fd = bufferHandler.nextFieldData();

        getSingleValue<<<1, 1>>>(ptr_test_read, *bufferHandler.fmd, fd);

        cudaDeviceSynchronize();

        std::cout << "ptr_test_read = " << std::fixed << std::setprecision(6) << *ptr_test_read << "\n";
    }
    
    // TODO: Write an example loop using buffering and measure it.

    // TODO: Free data properly in FieldData (maybe make an iterator)
    cudaFree(ptr_test_read);
    return 0;
}
