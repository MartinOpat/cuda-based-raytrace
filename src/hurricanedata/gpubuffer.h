#ifndef GPUBUFFER_H
#define GPUBUFFER_H

#include <string>
#include <memory>
#include <experimental/propagate_const>

#include "datareader.h"

struct DataHandle {
    float *d_data;
    size_t size;
};

class GPUBuffer {
public:
    static constexpr size_t numBufferedFiles = 3;

    GPUBuffer(DataReader& dataReader);

    void loadFile(size_t fileIndex, size_t bufferIndex); // Async call

    DataHandle getDataHandle(size_t bufferIndex); // Potentially blocking

    ~GPUBuffer();
private:
    class impl;
    std::experimental::propagate_const<std::unique_ptr<impl>> pImpl;
};

#endif //GPUBUFFER_H
