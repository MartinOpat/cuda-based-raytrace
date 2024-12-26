#ifndef GPUBUFFERHANDLER_H
#define GPUBUFFERHANDLER_H

#include "fielddata.h"
#include "gpubuffer.h"

#include <string>

class GPUBufferHandler {
public:
    GPUBufferHandler(GPUBuffer& gpuBuffer);

    FieldData nextFieldData();

    ~GPUBufferHandler();

    FieldMetadata *fmd;

    static constexpr size_t numberOfTimeStepsPerField = 2; // TODO: Move this to fielddata

    static void freeFieldData();

private:
    FieldData setupField(size_t endBufferInd);
    GPUBuffer& gpuBuffer;
    size_t fileInd;
    size_t bufferInd;
    size_t fieldInd;
    bool firstTimeStep = true;

    float **valArrays;
};

#endif //GPUBUFFERHANDLER_H
