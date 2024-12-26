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

    fmd->numberOfTimeStepsPerFile = 4; // TODO: Maybe find a better way to do this.
    fmd->timeSize = GPUBufferHandler::numberOfTimeStepsPerField;
}

FieldData GPUBufferHandler::setupField(size_t newEndBufferInd) {
    
    FieldData fd;
    cudaMallocManaged(&fd.valArrays, sizeof(sizeof(float *)*FieldData::FILESNUM));
    cudaMallocManaged(&fd.times, sizeof(sizeof(int *)*FieldData::FILESNUM));
    size_t fieldDataInd = 0;
    cout << "getting field from files " << bufferInd  << " to " << newEndBufferInd << "\n";
    for (int i = bufferInd; i <= newEndBufferInd; i++) {
        cout << "getting handle for " << i << "\n";
        DataHandle x = gpuBuffer.getDataHandle(i);
        fd.valArrays[fieldDataInd] = x.d_data;
        fd.times[fieldDataInd] = x.times;
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

    // size_t newBufferInd = (bufferInd + 1) % GPUBuffer::numBufferedFiles;
    // size_t newFieldInd = (fieldInd + ((bufferInd + 1) / 4)) % x.timeSize;

    // size_t endBufferInd = (bufferInd + GPUBufferHandler::numberOfTimeStepsPerField) % GPUBuffer::numBufferedFiles;
    // size_t endFieldInd = (fieldInd + ((bufferInd + GPUBufferHandler::numberOfTimeStepsPerField) / 4)) % x.timeSize;

    // size_t newEndBufferInd = (endBufferInd + 1) % GPUBuffer::numBufferedFiles;
    // size_t newEndFieldInd = (endFieldInd + ((endBufferInd + 1) / 4)) % x.timeSize;

    if(firstTimeStep) {
        firstTimeStep = false;
        return setupField(endFieldInd);
    } 

    if (newBufferInd != bufferInd) {
        fileInd++;
        gpuBuffer.loadFile(fileInd, bufferInd);
        bufferInd = newBufferInd;
        fieldInd = newFieldInd;
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
    cudaFree(fmd);
}