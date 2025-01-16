#ifndef MAT_H
#define MAT_H

#include "vec.h"
#include "consts.h"

__device__ float sampleVolumeNearest(float* volumeData, const int volW, const int volH, const int volD, int vx, int vy, int vz);
__device__ float sampleVolumeTrilinear(float* volumeData, const int volW, const int volH, const int volD, float fx, float fy, float fz);

__device__ Vec3 computeGradient(float* volumeData, const int volW, const int volH, const int volD, float fx, float fy, float fz);

__device__ unsigned int packUnorm4x8(float r, float g, float b, float a);

__device__ float clamp(float value, float min, float max);
__device__ float normalize(float value, float min, float max);

// Interpolate between two values
template <typename T>
__device__ T interpolate(T start, T end, float t) {
  return start + t * (end - start);
}



#endif // MAT_H
