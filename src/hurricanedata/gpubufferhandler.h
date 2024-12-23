#ifndef GPUBUFFERHANDLER_H
#define GPUBUFFERHANDLER_H

#include "fielddata.h"
#include "filepathmanager.h"

#include <string>

class GPUBufferHandler {
public:
    GPUBufferHandler(const std::string &path, std::string variableName);

    FieldData nextFieldData();

    ~GPUBufferHandler();

    FieldMetadata *fmd;

private:
    // TODO: Implement GPUBuffer
    FilePathManager filePathManager;
    std::string variableName;

    size_t presentTimeIndex;
    size_t fileIndex;
};

#endif //GPUBUFFERHANDLER_H
