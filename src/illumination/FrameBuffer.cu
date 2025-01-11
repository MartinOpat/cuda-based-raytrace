#include "FrameBuffer.h"
#include "linalg/linalg.h"


__host__ FrameBuffer::FrameBuffer(unsigned int w, unsigned int h) : w(w), h(h) {}

__device__ void FrameBuffer::writePixel(int x, int y, float r, float g, float b, float a) {
  int i = y * this->w + x;

  // the opengl buffer uses BGRA format; dunno why
	this->buffer[i] = packUnorm4x8(b, g, r, a);
}
