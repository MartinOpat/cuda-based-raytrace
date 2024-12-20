#ifndef GPUBUFFER_H
#define GPUBUFFER_H

#include "fielddata.h"

#include <string>

class GPUBuffer {
public:
    GPUBuffer(std::string path, std::string variableName);

    FieldData nextFieldData();

    ~GPUBuffer();

    FieldMetadata *fmd;
private:
    // TODO: Implement GPUBuffer
    std::string path;
    std::string variableName;

};

#endif //GPUBUFFER_H
