#ifndef FIELDDATA_H
#define FIELDDATA_H

#include <vector>

struct FieldMetadata {
    size_t widthSize; // Number of different longitudes
    size_t heightSize; // Number of different latitudes
    size_t depthSize; // Number of different levels

    // lons is a managed Unified Memory array of size widthCount that indicates 
    // that getVal(t, i, j, k) is a value with longitude of lons[i].
    // The other such arrays are similarly defined.
    double *lons; 
    double *lats;
    double *levs;
};

using FieldMetadata = FieldMetadata;

struct FieldData {
    static constexpr size_t FILESNUM = 2; // Number of files stored in a FieldData struct.

    // An array of length FILESNUM storing pointers to 4D arrays stored in device memory.
    float *valArrays[FILESNUM];

    size_t timeSize; // Number of different times
    // times is a managed Unified Memory array of size timeSize that indicates 
    // that getVal(md, d, t, i, j, k) is a value at time times[t].
    int *times;
};

using FieldData = FieldData;

extern __device__ float getVal(
    const FieldMetadata &md,
    const FieldData &d,
    const size_t &timeInd,
    const size_t &levInd,
    const size_t &latInd,
    const size_t &lonInd
);

#endif //FIELDDATA_H
