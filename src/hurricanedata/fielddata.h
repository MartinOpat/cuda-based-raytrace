#ifndef FIELDDATA_H
#define FIELDDATA_H

#include <vector>

struct FieldMetadata {
    size_t widthSize; // Number of different longitudes
    size_t heightSize; // Number of different latitudes
    size_t depthSize; // Number of different levels
    size_t timeSize; // Number of different times

    // lons is a managed Unified Memory array of size widthCount that indicates 
    // that getVal(t, i, j, k) is a value with longitude of lons[i].
    // The other such arrays are similarly defined.
    double *lons; 
    double *lats;
    double *levs;
    int *times;

    size_t numberOfTimeStepsPerFile;
};

using FieldMetadata = FieldMetadata;

/**
 * @brief Allows for accessing data of a time-slice of a scalar field
 * that may start in the middle of a file or go range over multiple files
 * by holding references to multiple files at a time.
 * 
 * @note Use the getVal method to index into it and get values.
 */
struct FieldData {
    static constexpr size_t FILESNUM = 2; // Number of files stored in a FieldData struct.
    static constexpr size_t numberOfTimeStepsPerField = 2;

    size_t fieldInd; // Indicates 

    // A uniform array of length FILESNUM storing pointers to 4D arrays stored in device memory.
    float **valArrays;

};

/**
 * @brief Get the scalar field value at a particular index (timeInd, levInd, latInd, lonInd).
 * Note that the timeInd may be counter-intuitive. See explanation in gpubufferhandler.h.
 * 
 * @return scalar field float value.
 */
extern __device__ float getVal(
    const FieldMetadata &md,
    const FieldData &d,
    const size_t &timeInd,
    const size_t &levInd,
    const size_t &latInd,
    const size_t &lonInd
);

#endif //FIELDDATA_H
