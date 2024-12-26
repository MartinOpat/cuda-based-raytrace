#include "gpubuffer.h"

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
    int *times;
    size_t timeSize;
};

struct LoadFileJob {
    size_t fileIndex;
    size_t bufferIndex;
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

using Job = variant<LoadFileJob, GetSizeJob, GetAxisIntJob, GetAxisDoubleJob>;

class GPUBuffer::impl {
public:
    impl(DataReader& dataReader);
    void loadFile(LoadFileJob job);
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
    void worker();
    queue<Job> jobs;
    unique_ptr<thread> ioworker;
    cudaStream_t iostream;
    DataReader& dataReader;
    condition_variable queuecv;
    mutex queuecv_m;
    bool workerRunning = true;
};

GPUBuffer::GPUBuffer(DataReader& dataReader): pImpl(make_unique<impl>(dataReader)) { }

size_t GPUBuffer::impl::getSize(size_t fileIndex) {
    promise<size_t> promise;
    future<size_t> future = promise.get_future();
    {
        lock_guard<mutex> lk(queuecv_m);
        jobs.push(GetSizeJob{fileIndex, move(promise)});
    }
    queuecv.notify_all();

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
    {
        lock_guard<mutex> lk(queuecv_m);
        jobs.push(GetAxisDoubleJob{fileIndex, axisName, move(promise)});
    }
    queuecv.notify_all();

    future.wait();

    return future.get();
}
template pair<size_t, double *> GPUBuffer::impl::getAxis<double>(size_t fileIndex, const string& axisName);

template <>
pair<size_t, int *> GPUBuffer::impl::getAxis(size_t fileIndex, const string& axisName) {
    promise<pair<size_t, int *>> promise;
    future<pair<size_t, int *>> future = promise.get_future();
    {
        lock_guard<mutex> lk(queuecv_m);
        jobs.push(GetAxisIntJob{fileIndex, axisName, move(promise)});
    }
    queuecv.notify_all();

    future.wait();

    return future.get();
}
template pair<size_t, int *> GPUBuffer::impl::getAxis<int>(size_t fileIndex, const string& axisName);

GPUBuffer::impl::impl(DataReader& dataReader): dataReader(dataReader) {
    cudaStreamCreate(&iostream);

    ioworker = make_unique<thread>([this]() { worker(); });

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
            cudaMallocManaged(&file.times, sizeof(int)*sizeTime);
            file.size = size;
            file.valid = false;
            file.timeSize = sizeTime;
        }
        // loadFile(i, i);
    }
}

GPUBuffer::~GPUBuffer() {
}

GPUBuffer::impl::~impl() {
    {
        lock_guard<mutex> lk(queuecv_m);
        workerRunning = false;
    }
    queuecv.notify_all();
    ioworker->join();
    for (size_t i = 0; i < numBufferedFiles; i++) {
        File &file = buffer[i];
        cudaFree(file.d_data);
        cudaFree(file.h_data);
        cudaFree(file.times);
    }
    cudaStreamDestroy(iostream);
}

void GPUBuffer::impl::loadFile(LoadFileJob job) {
    File &file = buffer[job.bufferIndex];

    {
        lock_guard<mutex> lk(file.m);
        assert(!file.valid);
        dataReader.loadFile<int>(file.times, job.fileIndex, "time"); // TODO: Times dont store anything useful :(
        cout << "times[1] (inside inside) " << file.times[1]  << " for file with fileindex = " << job.fileIndex <<  "\n";
        dataReader.loadFile<float>(file.h_data, job.fileIndex);
        cudaMemcpyAsync(file.d_data, file.h_data, sizeof(float)*file.size, cudaMemcpyHostToDevice, iostream);
        cudaStreamSynchronize(iostream);
        buffer[job.bufferIndex].valid = true;
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
    pImpl->loadFile(fileIndex, bufferIndex);
}

void GPUBuffer::impl::loadFile(size_t fileIndex, size_t bufferIndex) {
    LoadFileJob job = {
        .fileIndex = fileIndex,
        .bufferIndex = bufferIndex
    };

    // Main thread theoretically blocks on 2 mutexes here
    // but it _should_ never have to wait for them.
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
    {
        std::unique_lock<std::mutex> lk(queuecv_m, std::defer_lock);
        bool lockval = lk.try_lock();
        if (!lockval) {
            cout << "waited on IOworker queue during loadFile orchestration :(\n";
            lk.lock();
        }
        jobs.push(job);
    }
    queuecv.notify_all();
}

void GPUBuffer::impl::worker() {
    while(workerRunning) {
        Job job;
        {
            unique_lock<mutex> lk(queuecv_m);
            queuecv.wait(lk, [this]{ return !workerRunning || !jobs.empty(); });
            if (!workerRunning) {
                return;
            }

            job = move(jobs.front());
            jobs.pop();
        }
        if(holds_alternative<LoadFileJob>(job)) {
            loadFile(get<LoadFileJob>(job));
        } else if(holds_alternative<GetSizeJob>(job)) {
            getSize(move(get<GetSizeJob>(job)));
        } else if(holds_alternative<GetAxisDoubleJob>(job)) {
            getAxis(move(get<GetAxisDoubleJob>(job)));
        } else if(holds_alternative<GetAxisIntJob>(job)) {
            getAxis(move(get<GetAxisIntJob>(job)));
        }

    }
}

DataHandle GPUBuffer::getDataHandle(size_t bufferIndex) {
    File &file = pImpl->buffer[bufferIndex];

    // TODO: Might be nice to measure the blocking time here.
    unique_lock<mutex> lk(file.m);
    file.cv.wait(lk, [this, bufferIndex]{ return  pImpl->buffer[bufferIndex].valid; });

    DataHandle dh = {
        .d_data = file.d_data,
        .times = file.times,
        .timeSize = file.timeSize,
        .size = file.size
    };
    return dh;
}