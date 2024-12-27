#ifndef GPUBUFFERHANDLER_H
#define GPUBUFFERHANDLER_H

#include "fielddata.h"
#include "gpubuffer.h"

#include <string>

/**
 * @brief Responsible for deciding when the GPUBuffer should load and unload files.
 * Also assembles and gives access to FieldData.
 * 
 * You will need to interface with this class.
 */
class GPUBufferHandler {
public:
    GPUBufferHandler(GPUBuffer& gpuBuffer);

    /**
     * @brief Produces a FieldData which can be used to retrieve values for a time-slice
     * into a scalar field with a width of FieldData::numberOfTimeStepsPerField.
     * 
     * @details This method always increments the start point of the time-slice
     * by 1. See below:
     * 
     * time steps                    =  0  1  2  3  4  5  6  7  8  9  10  11  12 13 ...
     * files (4 time steps per file) = [0  1  2  3][4  5  6  7][8  9  10  11][12 13 ...
     * nextFieldData() (1st call)    = [0  1]
     * nextFieldData() (2nd call)    =    [1  2]
     * nextFieldData() (3rd call)    =       [2  3]
     * nextFieldData() (4th call)    =          [3  4]
     * nextFieldData() (5th call)    =             [4  5]
     * etc...
     *  
     * When getting values from a FieldData using the getVal method, 
     * the time index is relative to the start of the time-slice.
     * 
     * This means that for d = nextFieldData() (4th call),
     * getVal(.., fieldData=d, ..., timeInd = 1, ...) gives a value at 
     * absolute time step 5 as seen above.
     */
    FieldData nextFieldData();

    ~GPUBufferHandler();

    /**
     * You can get the FieldMetaData from here.
     */
    FieldMetadata *fmd;

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
