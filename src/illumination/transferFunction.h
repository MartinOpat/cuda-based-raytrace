#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "linalg/linalg.h"
#include "consts.h"
#include "shading.h"

// --------------------------- Color mapping ---------------------------


// --------------------------- Volume sampling ---------------------------
__device__ float sampleVolumeNearest(float* volumeData, const int volW, const int volH, const int volD, int vx, int vy, int vz);
__device__ float sampleVolumeTrilinear(float* volumeData, const int volW, const int volH, const int volD, float fx, float fy, float fz);


// --------------------------- Transfer function ---------------------------
__device__ float4 transferFunction(float density, const Vec3& grad, const Point3& pos, const Vec3& rayDir);


#endif // TRANSFER_FUNCTION_H