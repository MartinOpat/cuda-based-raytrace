#include "FrameBuffer.h"
#include "linalg/linalg.h"


__host__ FrameBuffer::FrameBuffer(unsigned int w, unsigned int h) : w(w), h(h) {
  this->buffer_size = w*h*sizeof(unsigned int);
  cudaMalloc((void**)&this->buffer, this->buffer_size);
  cudaMemset(this->buffer, 0, this->buffer_size);
}

__device__ void FrameBuffer::writePixel(int x, int y, float r, float g, float b) {
  int i = y * this->w + x;

  // the opengl buffer uses BGRA format; dunno why
	this->buffer[i] = packUnorm4x8(b, g, r, 1.0f);
}
