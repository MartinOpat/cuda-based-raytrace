#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include "cuda_runtime.h"
#include "linalg/linalg.h"
#include <cstdint>


class FrameBuffer {
public:
  unsigned int* buffer;
  std::size_t buffer_size;
  unsigned int w;
  unsigned int h;

  __host__ FrameBuffer(unsigned int w, unsigned int h);
  __device__ void writePixel(int x, int y, float r, float g, float b, float a);
};

#endif // FRAMEBUFFER_H
