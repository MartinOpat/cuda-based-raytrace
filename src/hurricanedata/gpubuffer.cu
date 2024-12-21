#include "gpubuffer.h"
#include "fielddata.h"
#include "gpubufferhelper.h"

#include <netcdf>


using namespace std;
using namespace netCDF;

GPUBuffer::GPUBuffer(std::string path, std::string variableName):
path(path), variableName(variableName) {
    NcFile data(path, NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    cudaMallocManaged(&fmd, sizeof(FieldMetadata));

    readAndAllocateAxis<double>(&fmd->lons, &fmd->widthSize, vars.find("lon")->second);
    readAndAllocateAxis<double>(&fmd->lats, &fmd->heightSize, vars.find("lat")->second);
    readAndAllocateAxis<double>(&fmd->levs, &fmd->depthSize, vars.find("lev")->second);
}

FieldData GPUBuffer::nextFieldData() {
    NcFile data(path, NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    FieldData fd;
    size_t timeSize;
    readAndAllocateAxis(&fd.times, &fd.timeSize, vars.find("time")->second);

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
    cudaError_t status = cudaMalloc(&fd.valArrays[0], sizeof(float)*length);
    if (status != cudaSuccess)
        cout << "Error allocating memory: " << status << "\n";

    cudaMemcpyAsync(fd.valArrays[0], h_array, sizeof(float)*length, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize(); // Heavy hammer synchronisation // TODO: Use streams

    cudaFreeHost(h_array);

    return fd;
}

GPUBuffer::~GPUBuffer() {
    cudaFree(fmd->lons);
    cudaFree(fmd->lats);
    cudaFree(fmd->levs);
    cudaFree(fmd);
}