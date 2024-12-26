#include "hurricanedata/fielddata.h"
#include "hurricanedata/gpubufferhandler.h"
#include "hurricanedata/datareader.h"
#include "hurricanedata/gpubuffer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <iomanip> 

__global__ void middleOfTwoValues(float *ans, const FieldMetadata &fmd, FieldData fd) {
    float xi = getVal(fmd, fd, 0, 20, 100, 100);
    float yi = getVal(fmd, fd, 1, 20, 100, 100);
    *ans = (xi+yi)/2;
}

int main() {
    std::string path = "data/atmosphere_MERRA-wind-speed[179253532]";

    std::string variable = "T";

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

        middleOfTwoValues<<<1, 1>>>(ptr_test_read, *bufferHandler.fmd, fd);

        cudaDeviceSynchronize();

        std::cout << "ptr_test_read = " << std::fixed << std::setprecision(6) << *ptr_test_read << "\n";
    }
    
    // TODO: measure data transfer time in this example code.
    cudaFree(ptr_test_read);
    return 0;
}
