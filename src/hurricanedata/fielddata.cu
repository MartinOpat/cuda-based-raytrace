#include "fielddata.h"

__device__ float getVal(
    const FieldMetadata &md,
    const FieldData &d,
    const size_t &timeInd,
    const size_t &levInd,
    const size_t &latInd,
    const size_t &lonInd
) {
    size_t sizeSpatialData = md.widthSize*md.heightSize*md.depthSize;
    size_t size2DMapData = md.widthSize*md.heightSize;
    return d.valArrays[0][
        timeInd*sizeSpatialData
        + levInd*size2DMapData
        + latInd*md.widthSize
        + lonInd
    ]; 
}