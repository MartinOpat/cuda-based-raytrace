#ifndef WORKER_H
#define WORKER_H

#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <future>

class Worker {
public:
    using Job = std::packaged_task<void()>;
    Worker();
    void enqueueJob(Job job);
    void stop();
    ~Worker();
private:
    void jobRunner();
    bool workerRunning;
    std::queue<Job> jobs;
    std::condition_variable queuecv;
    std::mutex queuecv_m;
    std::thread ioworker;
};

#endif //WORKER_H
