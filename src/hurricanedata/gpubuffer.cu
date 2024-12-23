#include "gpubuffer.h"

#include <mutex>
#include <thread>
#include <queue>
#include <cassert>
#include <condition_variable>
#include <iostream>

#include <netcdf>
using namespace netCDF;

using namespace std;

struct File {
    condition_variable cv;
    mutex m;
    bool valid;
    size_t size;
    float *h_data; // host data
    float *d_data; // device data
};

struct LoadFileJob {
    size_t fileIndex;
    size_t bufferIndex;
};

class GPUBuffer::impl {
public:
    impl(DataReader& dataReader);
    void loadFile(LoadFileJob job);
    ~impl();

    File buffer[numBufferedFiles];

    // Thread worker things
    void worker();
    queue<LoadFileJob> jobs;
    unique_ptr<thread> ioworker;
    cudaStream_t iostream;
    DataReader& dataReader;
    condition_variable queuecv;
    mutex queuecv_m;
    bool workerRunning = true;
};

GPUBuffer::GPUBuffer(DataReader& dataReader): pImpl(make_unique<impl>(dataReader)) { }

GPUBuffer::impl::impl(DataReader& dataReader): dataReader(dataReader) {
    cudaStreamCreate(&iostream);
    // size_t size = dataReader.fileLength(0);
    size_t size = 5;

    for (size_t i = 0; i < numBufferedFiles; i++) {
        {
            File &file = buffer[i];
            lock_guard<mutex> lk(file.m);
            cudaMallocHost(&file.h_data, sizeof(float)*size);
            cudaError_t status = cudaMalloc(&file.d_data, sizeof(float)*size);
            if (status != cudaSuccess)
                cerr << "Error allocating device memory. Status code: " << status << "\n";
            file.size = size;
            file.valid = false;
        }
        {
            lock_guard<mutex> lk(queuecv_m);
            LoadFileJob job = {
                .fileIndex = i,
                .bufferIndex = i
            };
            jobs.push(job);
        }

    }
    auto t = thread([]() {
        NcFile data("data/MERRA2_400.inst6_3d_ana_Np.20120911.nc4", NcFile::read);
        multimap<string, NcVar> vars = data.getVars();
    });
    t.join();

    auto tt = thread([]() {
        NcFile data("data/MERRA2_400.inst6_3d_ana_Np.20120911.nc4", NcFile::read);
        multimap<string, NcVar> vars = data.getVars();
    });
    tt.join();

    ioworker = make_unique<thread>([this]() { worker(); });
}

GPUBuffer::~GPUBuffer() { }

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
    }
    cudaStreamDestroy(iostream);
}

void GPUBuffer::impl::loadFile(LoadFileJob job) {
    File &file = buffer[job.bufferIndex];

    {
        lock_guard<mutex> lk(file.m);
        assert(!file.valid);
        dataReader.loadFile<float>(file.h_data, job.fileIndex);
        cudaMemcpyAsync(file.d_data, file.h_data, sizeof(float)*file.size, cudaMemcpyHostToDevice, iostream);
        cudaStreamSynchronize(iostream);
        buffer[job.bufferIndex].valid = true;
        cout << "loaded file with index" << job.bufferIndex << "\n";
    }
    file.cv.notify_all();
}

void GPUBuffer::loadFile(size_t fileIndex, size_t bufferIndex) {
    LoadFileJob job = {
        .fileIndex = fileIndex,
        .bufferIndex = bufferIndex
    };

    // Main thread theoretically blocks on 2 mutexes here
    // but it _should_ never have to wait for them.
    {
        File &file = pImpl->buffer[bufferIndex];

        std::unique_lock<std::mutex> lk(file.m, std::defer_lock);
        bool lockval = lk.try_lock();
        if (!lockval) {
            cout << "waited on GPUBuffer during loadFile orchestration :(\n";
            lk.lock();
        }
        file.valid = false;
    }
    {
        std::unique_lock<std::mutex> lk(pImpl->queuecv_m, std::defer_lock);
        bool lockval = lk.try_lock();
        if (!lockval) {
            cout << "waited on IOworker queue during loadFile orchestration :(\n";
            lk.lock();
        }
        pImpl->jobs.push(job);
    }
    pImpl->queuecv.notify_all();
}

void GPUBuffer::impl::worker() {
    while(workerRunning) {
        LoadFileJob job;
        {
            unique_lock<mutex> lk(queuecv_m);
            queuecv.wait(lk, [this]{ return !workerRunning || !jobs.empty(); });
            if (!workerRunning) {
                return;
            }

            job = jobs.front();
            jobs.pop();
        }
        loadFile(job);
    }
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