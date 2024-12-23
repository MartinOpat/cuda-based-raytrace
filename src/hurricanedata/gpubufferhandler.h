#ifndef GPUBUFFERHANDLER_H
#define GPUBUFFERHANDLER_H

#include "fielddata.h"
#include "gpubuffer.h"

#include <string>

class GPUBufferHandler {
public:
    GPUBufferHandler();

    FieldData nextFieldData();

    ~GPUBufferHandler();

    FieldMetadata *fmd;

private:
    GPUBuffer 
};

#endif //GPUBUFFERHANDLER_H
