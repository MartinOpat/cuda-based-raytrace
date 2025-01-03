#ifndef SHADING_H
#define SHADING_H

#include "linalg/linalg.h"
#include "consts.h"

__device__ Vec3 phongShading(const Vec3& normal, const Vec3& lightDir, const Vec3& viewDir, const Vec3& baseColor);


#endif // SHADING_H
