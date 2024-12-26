#include "gpubufferhandler.h"
#include "fielddata.h"

#include <iostream>

using namespace std;

GPUBufferHandler::GPUBufferHandler(GPUBuffer& gpuBuffer):
gpuBuffer(gpuBuffer), fieldInd(0), bufferInd(0), fileInd(0) {
    cudaMallocManaged(&fmd, sizeof(FieldMetadata));

    auto [widthSize, lons] = gpuBuffer.getAxis<double>(0, "lon");
    fmd->widthSize = widthSize;
    fmd->lons = lons;

    auto [heightSize, lats] = gpuBuffer.getAxis<double>(0, "lat");
    fmd->heightSize = heightSize;
    fmd->lats = lats;

    auto [depthSize, levs] = gpuBuffer.getAxis<double>(0, "lev");
    fmd->depthSize = depthSize;
    fmd->levs = levs;

    for (size_t i = 0; i < GPUBuffer::numBufferedFiles; i++) {
        gpuBuffer.loadFile(fileInd,fileInd);
        fileInd++;
    }

    fmd->timeSize = GPUBufferHandler::numberOfTimeStepsPerField;

    cudaMallocManaged(&fmd->times, sizeof(fmd->numberOfTimeStepsPerFile*sizeof(int)));

    auto [numberOfTimeStepsPerFile, times] = gpuBuffer.getAxis<int>(0, "time");
    fmd->numberOfTimeStepsPerFile = numberOfTimeStepsPerFile;
    fmd->times = times;

    cudaMallocManaged(&valArrays, sizeof(float *)*FieldData::FILESNUM);
}

FieldData GPUBufferHandler::setupField(size_t newEndBufferInd) {
    
    FieldData fd;
    size_t fieldDataInd = 0;
    fd.valArrays = valArrays;
    cout << "getting field from files " << bufferInd  << " to " << newEndBufferInd << " with a field index of " << fieldInd << "\n";
    for (int i = bufferInd; i <= newEndBufferInd; i++) {
        DataHandle x = gpuBuffer.getDataHandle(i);
        fd.valArrays[fieldDataInd] = x.d_data;
        fieldDataInd++;
    }
    fd.fieldInd = fieldInd;

    return fd;
}

FieldData GPUBufferHandler::nextFieldData() {
    DataHandle x = gpuBuffer.getDataHandle(bufferInd);
    size_t newFieldInd = (fieldInd + 1) % fmd->numberOfTimeStepsPerFile;
    size_t newBufferInd = (bufferInd + ((fieldInd + 1) / fmd->numberOfTimeStepsPerFile)) % GPUBuffer::numBufferedFiles;

    size_t endFieldInd = (fieldInd + GPUBufferHandler::numberOfTimeStepsPerField - 1) % fmd->numberOfTimeStepsPerFile;
    size_t endBufferInd = (bufferInd + (fieldInd + GPUBufferHandler::numberOfTimeStepsPerField - 1)/fmd->numberOfTimeStepsPerFile) % GPUBuffer::numBufferedFiles;

    size_t newEndFieldInd = (endFieldInd + 1) % fmd->numberOfTimeStepsPerFile;
    size_t newEndBufferInd = (endBufferInd + ((endFieldInd + 1) / fmd->numberOfTimeStepsPerFile)) % GPUBuffer::numBufferedFiles;

    if(firstTimeStep) {
        firstTimeStep = false;
        return setupField(endBufferInd);
    } 

    fieldInd = newFieldInd;

    if (newBufferInd != bufferInd) {
        fileInd++;
        gpuBuffer.loadFile(fileInd, bufferInd);
        bufferInd = newBufferInd;
    }

    if (newEndBufferInd != endBufferInd) {
        // maybe dont do things?
    }

    return setupField(newEndBufferInd);
}

GPUBufferHandler::~GPUBufferHandler() {
    cudaFree(fmd->lons);
    cudaFree(fmd->lats);
    cudaFree(fmd->levs);
    cudaFree(valArrays);
    cudaFree(fmd);
}