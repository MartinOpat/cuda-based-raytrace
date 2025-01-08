#include "worker.h"
#include <iostream>

using namespace std;

Worker::Worker(): workerRunning(false) {
    ioworker = thread([this]() { jobRunner(); });
}

void Worker::enqueueJob(Job job) {
    {
        lock_guard<mutex> lk(queuecv_m);
        jobs.push(move(job));
    }
    queuecv.notify_all();
}

void Worker::jobRunner() {
    workerRunning = true;
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

        job();
    }
}

void Worker::stop() {
    if (workerRunning == false) return;
    {
        lock_guard<mutex> lk(queuecv_m);
        workerRunning = false;
    }
    queuecv.notify_all();
    ioworker.join();
}

Worker::~Worker() {
    stop();
}