#ifndef RAYCASTER_H
#define RAYCASTER_H

// #include "Camera.h"
#include "cuda_runtime.h"
#include "FrameBuffer.h"
#include "linalg/linalg.h"

// #include <thrust/device_ptr.h>

__global__ void raycastKernel(float* volumeData, unsigned char* framebuffer);

struct Raycaster {

    // thrust::device_ptr<Camera*> d_camera;
    // CameraInfo camera_info;

    cudaGraphicsResource_t resources;
    FrameBuffer* fb;

    int w;
    int h;

    Raycaster() {};
    Raycaster(cudaGraphicsResource_t resources, int nx, int ny);
    // ~Raycaster();

    void set_camera(Vec3 position, Vec3 forward, Vec3 up);
    void render();
    void resize(int nx, int ny);
    // void raycastKernel(float* volumeData, unsigned char* framebuffer); // TODO: proper framebuffer class
};
#endif // RAYCASTER_H
