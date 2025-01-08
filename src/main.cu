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

    DataReader dataReaderU{path, "U"};

    DataReader dataReaderV{path, "V"};

    std::cout << "created datareader\n";

    GPUBuffer bufferU (dataReaderU);

    GPUBuffer bufferV (dataReaderV);

    std::cout << "created buffer\n";

    GPUBufferHandler bufferHandlerU(bufferU);

    GPUBufferHandler bufferHandlerV(bufferV);

    float *ptr_test_read;
    cudaMallocManaged(&ptr_test_read, sizeof(float));

    std::cout << "created buffer handler\n";
    for (int i = 0; i < 20; i++) {
        FieldData fdU = bufferHandlerU.nextFieldData();
        FieldData fdV = bufferHandlerV.nextFieldData();

        middleOfTwoValues<<<1, 1>>>(ptr_test_read, *bufferHandlerU.fmd, fdU);

        cudaDeviceSynchronize();
        std::cout << "ptr_test_read U = " << std::fixed << std::setprecision(6) << *ptr_test_read << "\n";

        middleOfTwoValues<<<1, 1>>>(ptr_test_read, *bufferHandlerV.fmd, fdV);

        cudaDeviceSynchronize();
        std::cout << "ptr_test_read V = " << std::fixed << std::setprecision(6) << *ptr_test_read << "\n";
    }
    
    // TODO: measure data transfer time in this example code.
    cudaFree(ptr_test_read);
    return 0;
}
