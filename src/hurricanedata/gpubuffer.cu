#include "gpubuffer.h"
#include "worker.h"

#include <mutex>
#include <thread>
#include <queue>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <future>
#include <variant>

using namespace std;

struct File {
    condition_variable cv;
    mutex m;
    bool valid;
    size_t size;
    float *h_data; // host data
    float *d_data; // device data
};

struct GetSizeJob {
    size_t fileIndex;
    promise<size_t> size;
};

struct GetAxisDoubleJob {
    size_t fileIndex;
    string axisName;
    promise<pair<size_t, double *>> axis;
};

struct GetAxisIntJob {
    size_t fileIndex;
    string axisName;
    promise<pair<size_t, int *>> axis;
};

class GPUBuffer::impl {
public:
    impl(DataReader& dataReader);
    void enqueueLoadFileJob(size_t fileIndex, size_t bufferIndex);
    void loadFile(size_t fileIndex, size_t bufferIndex);
    void getSize(GetSizeJob job);
    size_t getSize(size_t fileIndex); // Most probably blocking
    void getAxis(GetAxisDoubleJob job);
    void getAxis(GetAxisIntJob job);

    template <typename T>
    std::pair<size_t, T *> getAxis(size_t fileIndex, const std::string& axisName); // Most probably blocking
    ~impl();

    File buffer[numBufferedFiles];

    // Thread worker things
    cudaStream_t iostream; // TODO: Make this static?
    DataReader& dataReader;
    static Worker worker;
};

Worker GPUBuffer::impl::worker;

GPUBuffer::GPUBuffer(DataReader& dataReader): pImpl(make_unique<impl>(dataReader)) { }

size_t GPUBuffer::impl::getSize(size_t fileIndex) {
    promise<size_t> promise;
    future<size_t> future = promise.get_future();

    // worker.enqueueJob(std::make_unique<std::function<void()>>(
    //     [this, fileIndex, promise = move(promise)]() mutable {
    //         getSize(GetSizeJob{fileIndex, move(promise)});
    //     })
    // );
    auto task = std::packaged_task<void()>(
        [this, fileIndex, promise = move(promise)]() mutable {
            getSize(GetSizeJob{fileIndex, move(promise)});
        }
    );
    worker.enqueueJob(move(task));

    future.wait();

    return future.get();
}

template <typename T>
pair<size_t, T *> GPUBuffer::getAxis(size_t fileIndex, const string& axisName) {
    return pImpl->getAxis<T>(fileIndex, axisName);
}
template pair<size_t, int *> GPUBuffer::getAxis<int>(size_t fileIndex, const string& axisName);
template pair<size_t, double *> GPUBuffer::getAxis<double>(size_t fileIndex, const string& axisName);

template <>
pair<size_t, double *> GPUBuffer::impl::getAxis(size_t fileIndex, const string& axisName) {
    promise<pair<size_t, double *>> promise;
    future<pair<size_t, double *>> future = promise.get_future();


    // worker.enqueueJob(std::make_unique<std::function<void()>>(
    //     [this, fileIndex, axisName, promise = move(promise)]() mutable {
    //         getAxis(GetAxisDoubleJob{fileIndex, axisName, move(promise)}); 
    //     })
    // );
    // worker.enqueueJob([this, fileIndex, axisName, promise = move(promise)]() mutable { getAxis(GetAxisDoubleJob{fileIndex, axisName, move(promise)}); });
    auto task = std::packaged_task<void()>(
        [this, fileIndex, axisName, promise = move(promise)]() mutable {
            getAxis(GetAxisDoubleJob{fileIndex, axisName, move(promise)}); 
        }
    );
    worker.enqueueJob(move(task));

    future.wait();

    return future.get();
}
template pair<size_t, double *> GPUBuffer::impl::getAxis<double>(size_t fileIndex, const string& axisName);

template <>
pair<size_t, int *> GPUBuffer::impl::getAxis(size_t fileIndex, const string& axisName) {
    promise<pair<size_t, int *>> promise;
    future<pair<size_t, int *>> future = promise.get_future();
    auto task = std::packaged_task<void()>(
        [this, fileIndex, axisName, promise = move(promise)]() mutable { 
            getAxis(GetAxisIntJob{fileIndex, axisName, move(promise)});
        }
    );
    worker.enqueueJob(move(task));

    future.wait();

    return future.get();
}
template pair<size_t, int *> GPUBuffer::impl::getAxis<int>(size_t fileIndex, const string& axisName);

GPUBuffer::impl::impl(DataReader& dataReader): dataReader(dataReader) {
    cudaStreamCreate(&iostream);

    size_t size = getSize(0);
    auto x = getAxis<int>(0, "time");
    size_t sizeTime = x.first;
    cudaFree(x.second);
    for (size_t i = 0; i < numBufferedFiles; i++) {
        {
            File &file = buffer[i];
            lock_guard<mutex> lk(file.m);
            cudaMallocHost(&file.h_data, sizeof(float)*size);
            cudaMalloc(&file.d_data, sizeof(float)*size);
            file.size = size;
            file.valid = false;
        }
    }
}

GPUBuffer::~GPUBuffer() {
}

GPUBuffer::impl::~impl() {
    worker.stop();
    for (size_t i = 0; i < numBufferedFiles; i++) {
        File &file = buffer[i];
        cudaFree(file.d_data);

        cudaFreeHost(file.h_data);
    }
    cudaStreamDestroy(iostream);
}

void GPUBuffer::impl::loadFile(size_t fileIndex, size_t bufferIndex) {
    File &file = buffer[bufferIndex];
    {
        lock_guard<mutex> lk(file.m);
        cout << "loading file with index " << fileIndex << "\n";
        assert(!file.valid);
        dataReader.loadFile<float>(file.h_data, fileIndex);
        cudaMemcpyAsync(file.d_data, file.h_data, sizeof(float)*file.size, cudaMemcpyHostToDevice, iostream);
        cudaStreamSynchronize(iostream);
        buffer[bufferIndex].valid = true;
    }
    file.cv.notify_all();
}

void GPUBuffer::impl::getSize(GetSizeJob job) {
    size_t size = dataReader.fileLength(job.fileIndex);
    job.size.set_value(size);
}

void GPUBuffer::impl::getAxis(GetAxisDoubleJob job) {
    pair<size_t, double *> array;
    array.first = dataReader.axisLength(job.fileIndex, job.axisName);
    cudaError_t status = cudaMallocManaged(&array.second, array.first*sizeof(double));
    dataReader.loadFile<double>(array.second, job.fileIndex, job.axisName);
    job.axis.set_value(array);
}

void GPUBuffer::impl::getAxis(GetAxisIntJob job) {
    pair<size_t, int *> array;
    array.first = dataReader.axisLength(job.fileIndex, job.axisName);
    cudaError_t status = cudaMallocManaged(&array.second, array.first*sizeof(int));
    dataReader.loadFile<int>(array.second, job.fileIndex, job.axisName);
    job.axis.set_value(array);
}

void GPUBuffer::loadFile(size_t fileIndex, size_t bufferIndex) {
    pImpl->enqueueLoadFileJob(fileIndex, bufferIndex);
}

void GPUBuffer::impl::enqueueLoadFileJob(size_t fileIndex, size_t bufferIndex) {
    {
        File &file = buffer[bufferIndex];

        std::unique_lock<std::mutex> lk(file.m, std::defer_lock);
        bool lockval = lk.try_lock();
        if (!lockval) {
            cout << "waited on GPUBuffer during loadFile orchestration :(\n";
            lk.lock();
        }
        file.valid = false;
    }

    auto task = std::packaged_task<void()>(bind(
        &GPUBuffer::impl::impl::loadFile,
        this,
        fileIndex,
        bufferIndex
    ));
    worker.enqueueJob(move(task));
    // worker.enqueueJob([this, fileIndex, bufferIndex](){ loadFile(fileIndex, bufferIndex); });
}

DataHandle GPUBuffer::getDataHandle(size_t bufferIndex) {
    File &file = pImpl->buffer[bufferIndex];

    // TODO: Might be nice to measure the blocking time here.
    unique_lock<mutex> lk(file.m);
    file.cv.wait(lk, [this, bufferIndex]{ return  pImpl->buffer[bufferIndex].valid; });

    DataHandle dh = {
        .d_data = file.d_data,
        .size = file.size
    };
    return dh;
}