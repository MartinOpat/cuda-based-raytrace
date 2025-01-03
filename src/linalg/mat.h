#ifndef MAT_H
#define MAT_H

#include "vec.h"

__device__ Vec3 computeGradient(float* volumeData, const int volW, const int volH, const int volD, int x, int y, int z);

__device__ unsigned int packUnorm4x8(float r, float g, float b, float a);

#endif // MAT_H
