#ifndef RAYCASTER_H
#define RAYCASTER_H

#include "cuda_runtime.h"
#include "FrameBuffer.h"
#include "linalg/linalg.h"

// #include <thrust/device_ptr.h>

__global__ void raycastKernel(float* volumeData, unsigned char* framebuffer);

struct Raycaster {

  cudaGraphicsResource_t resources;
  FrameBuffer* fb;
  float* data;

  int w;
  int h;
  

  Raycaster(cudaGraphicsResource_t resources, int nx, int ny, float* data);
  // ~Raycaster();

  void set_camera(Vec3 position, Vec3 forward, Vec3 up);
  void render();
  void resize(int nx, int ny);
};
#endif // RAYCASTER_H
