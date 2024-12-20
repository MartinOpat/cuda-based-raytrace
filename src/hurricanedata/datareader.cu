#include "datareader.h"

#include <netcdf>

using namespace std;
using namespace netCDF;

std::vector<float> readData(std::string path, std::string variableName) {
    netCDF::NcFile data(path, netCDF::NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(variableName)->second;   

    int length = 1;
    for (NcDim dim: var.getDims()) {
        length *= dim.getSize();
    }

    vector<float> vec(length);

    var.getVar(vec.data());

    return vec;
}

std::pair<float*, size_t> loadDataToDevice(std::string path, std::string variableName) {
    netCDF::NcFile data(path, netCDF::NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(variableName)->second;   

    int length = 1;
    for (NcDim dim: var.getDims()) {
        length *= dim.getSize();
    }

    // Store NetCDF variable in pinned memory on host
    float *h_array;

    cudaMallocHost(&h_array, sizeof(float)*length);

    var.getVar(h_array);

    // Copy data to device
    float *d_array;

    cudaError_t status = cudaMalloc(&d_array, sizeof(float)*length);
    if (status != cudaSuccess)
        cout << "Error allocating memory: " << status << "\n";

    cudaMemcpyAsync(d_array, h_array, sizeof(float)*length, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize(); // Heavy hammer synchronisation // TODO: Use streams

    cudaFreeHost(h_array);

    return std::pair(d_array, length);
}