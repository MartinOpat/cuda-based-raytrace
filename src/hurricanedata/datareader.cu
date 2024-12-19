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

struct cudaArray* loadDataToDevice(std::string path, std::string variableName) {
    netCDF::NcFile data(path, netCDF::NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(variableName)->second;   

    struct cudaChannelFormatDesc arrayType = {
        .x = 32,
        .y = 0,
        .z = 0,
        .w = 0,
        .f = cudaChannelFormatKindFloat
    }; // Float-32
    
    struct cudaExtent extent = {
        .width = var.getDim(3).getSize(), // longitude
        .height = var.getDim(2).getSize(), // latitude
        .depth = var.getDim(1).getSize(), // level
    };

    struct cudaArray *array;

    cudaError_t error = cudaMalloc3DArray(&array, &arrayType, extent, 0);
    cout << "cuda error: " << error << "\n";

    return array;
}