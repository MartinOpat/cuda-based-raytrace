#ifndef TRANSFER_FUNCTION_H
#define TRANSFER_FUNCTION_H

#include "linalg/linalg.h"
#include "consts.h"
#include "shading.h"

// --------------------------- Color mapping ---------------------------

// --------------------------- Transfer function ---------------------------
__device__ float4 transferFunction(float density, const Vec3& grad, const Point3& pos, const Vec3& rayDir);


#endif // TRANSFER_FUNCTION_H