#pragma once

#include <cuda_runtime.h>
#include <cmath>

#include "linalg/linalg.h"

// TODO: This is technically just for debugging, but if it is to be used outside of that, it should be a made into a proper class (I mean, just look at those functions below, it screams "add class attributes")
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

// A function to generate two concentric spherical shells
__host__ void generateVolume(float* volumeData, int volW, int volH, int volD) {
    int cx = volW / 2;
    int cy = volH / 2;
    int cz = volD / 2;
    float maxRadius = static_cast<float>(volW) * 0.5f;

    // Two shells
    float shell1Inner = 0.2f * maxRadius;
    float shell1Outer = 0.3f * maxRadius;
    float shell2Inner = 0.4f * maxRadius;
    float shell2Outer = 0.5f * maxRadius;

    float shell1Intensity = 0.8f;
    float shell2Intensity = 0.6f;

    for (int z = 0; z < volD; ++z) {
        for (int y = 0; y < volH; ++y) {
            for (int x = 0; x < volW; ++x) {
                float dx = (float)(x - cx);
                float dy = (float)(y - cy);
                float dz = (float)(z - cz);
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                float intensity = 0.0f;

                // Shell 1
                if (dist >= shell1Inner && dist <= shell1Outer) {
                    float mid = 0.5f * (shell1Inner + shell1Outer);
                    if (dist < mid) {
                        float inFactor = (dist - shell1Inner) / (mid - shell1Inner);
                        intensity += shell1Intensity * inFactor;
                    } else {
                        float outFactor = (shell1Outer - dist) / (shell1Outer - mid);
                        intensity += shell1Intensity * outFactor;
                    }
                }

                // Shell 2
                if (dist >= shell2Inner && dist <= shell2Outer) {
                    float mid = 0.5f * (shell2Inner + shell2Outer);
                    if (dist < mid) {
                        float inFactor = (dist - shell2Inner) / (mid - shell2Inner);
                        intensity += shell2Intensity * inFactor;
                    } else {
                        float outFactor = (shell2Outer - dist) / (shell2Outer - mid);
                        intensity += shell2Intensity * outFactor;
                    }
                }

                if (intensity > 1.0f) intensity = 1.0f;
                    volumeData[z * volW * volH + y * volW + x] = intensity;
            }
        }
    }
}

// Samples the voxel nearest to the given coordinates. TODO: Can be re-used in other places
__device__ float sampleVolumeNearest(float* volumeData, const int volW, const int volH, const int volD, int vx, int vy, int vz) {
    if (vx < 0) vx = 0;
    if (vy < 0) vy = 0;
    if (vz < 0) vz = 0;
    if (vx >= volW) vx = volW  - 1;
    if (vy >= volH) vy = volH - 1;
    if (vz >= volD) vz = volD  - 1;

    int idx = vz * volW * volH + vy * volD + vx;
    return volumeData[idx];
}