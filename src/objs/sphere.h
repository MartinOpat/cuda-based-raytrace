#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "linalg/linalg.h"

struct Sphere {
    Vec3 center;
    double radius;
    Vec3 color;

    __device__ bool intersect(const Vec3& rayOrigin, const Vec3& rayDir, double& t) const {
        Vec3 oc = rayOrigin - center;
        double b = oc.dot(rayDir);
        double c = oc.dot(oc) - radius * radius;
        double h = b * b - c;
        if (h < 0.0) return false;
        h = sqrt(h);
        t = -b - h;
        return true;
    }
};