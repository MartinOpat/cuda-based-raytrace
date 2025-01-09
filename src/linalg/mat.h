#ifndef MAT_H
#define MAT_H

#include "vec.h"

__device__ Vec3 computeGradient(float* volumeData, const int volW, const int volH, const int volD, int x, int y, int z);

__device__ unsigned int packUnorm4x8(float r, float g, float b, float a);

__device__ float clamp(float value, float min, float max);
__device__ float normalize(float value, float min, float max);

template <typename T>
__device__ float interpolate(T start, T end, float t);


#endif // MAT_H
