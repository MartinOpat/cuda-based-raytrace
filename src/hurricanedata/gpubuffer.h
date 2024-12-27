#ifndef GPUBUFFER_H
#define GPUBUFFER_H

#include <string>
#include <memory>
#include <experimental/propagate_const>

#include "datareader.h"

struct DataHandle {
    float *d_data; // Device memory
    size_t size;
    // Could include the data stored in host memory h_data in this handle if it were needed.
};

/**
 * @brief Handles the asynchronous (un)loading data to the GPU. The rest of the
 * application should not have to interface directly with this class. Getting data
 * should go over the GPUBufferHandler class.
 * 
 * @note This class uses a queue to give loading jobs to the worker thread.
 * NetCDF-C is not thread safe so you may never read data using netCDF directly
 * on any other thread than this worker thread.
 * 
 * Assumes all the data in the nc4 files is the same size.
 * 
 */
class GPUBuffer {
public:
    static constexpr size_t numBufferedFiles = 3; // Number of files stored in memory at one time.

    GPUBuffer(DataReader& dataReader);

    /**
     * @brief Asynchronously tells the GPUBuffer to eventually load a particular file index
     * into a buffer index (in which part of the buffer the file should be stored).
     */
    void loadFile(size_t fileIndex, size_t bufferIndex); // No blocking

    /**
     * @brief Get the values stored in a particular buffer index
     * @return A struct DataHandle that points to the memory and gives its size.
     */
    DataHandle getDataHandle(size_t bufferIndex); // Potentially blocking

    /**
     * @brief Get the data from an axis (e.g. longitude) and its size.
     * @note This is a blocking operation, so it makes a job for the worker
     * to read the data and then waits untill the job is completed.
     */
    template <typename T>
    std::pair<size_t, T *> getAxis(size_t fileIndex, const std::string& axisName); // Most probably blocking

    ~GPUBuffer();
private:
    class impl;
    std::experimental::propagate_const<std::unique_ptr<impl>> pImpl;
};

#endif //GPUBUFFER_H
