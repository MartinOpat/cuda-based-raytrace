#ifndef RAYCASTER_H
#define RAYCASTER_H

#include <cuda_runtime.h>
#include "linalg/linalg.h"
#include "consts.h"
#include "shading.h"



// Raycast + phong, TODO: Consider wrapping in a class
__global__ void raycastKernel(float* volumeData, unsigned char* framebuffer, int d_volumeWidth, int d_volumeHeight, int d_volumeDepth) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMAGE_WIDTH || py >= IMAGE_HEIGHT) return;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;

    // Multiple samples per pixel
    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
        // Map to [-1, 1]
        float u = ((px + 0.5f) / IMAGE_WIDTH ) * 2.0f - 1.0f;
        float v = ((py + 0.5f) / IMAGE_HEIGHT) * 2.0f - 1.0f;

        // TODO: Move this (and all similar transformation code) to its own separate file
        float tanHalfFov = tanf(fov * 0.5f);
        u *= tanHalfFov;
        v *= tanHalfFov;

        // Find ray direction
        Vec3 cameraRight = (cameraDir.cross(cameraUp)).normalize();
        cameraUp = (cameraRight.cross(cameraDir)).normalize();
        Vec3 rayDir = (cameraDir + cameraRight*u + cameraUp*v).normalize();

        // Intersect (for simplicity just a 3D box from 0 to 1 in all dimensions) - TODO: Think about whether this is the best way to do this
        float tNear = 0.0f;
        float tFar  = 1e6f;
        auto intersectAxis = [&](float start, float dirVal) {
            if (fabsf(dirVal) < epsilon) {
                if (start < 0.f || start > 1.f) {
                    tNear = 1e9f;
                    tFar  = -1e9f;
                }
            } else {
                float t0 = (0.0f - start) / dirVal;
                float t1 = (1.0f - start) / dirVal;
                if (t0>t1) { 
                    float tmp=t0; 
                    t0=t1; 
                    t1=tmp; 
                }
                if (t0>tNear) tNear = t0;
                if (t1<tFar ) tFar  = t1;
            }
        };

        intersectAxis(cameraPos.x, rayDir.x);
        intersectAxis(cameraPos.y, rayDir.y);
        intersectAxis(cameraPos.z, rayDir.z);

        if (tNear > tFar) continue;  // No intersectionn
        if (tNear < 0.0f) tNear = 0.0f;

        float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f;
        float alphaAccum = 0.0f;

        float tCurrent = tNear;
        while (tCurrent < tFar && alphaAccum < alphaAcumLimit) {
            Point3 pos = cameraPos + rayDir * tCurrent;

            // Convert to volume indices
            float fx = pos.x * (d_volumeWidth  - 1);
            float fy = pos.y * (d_volumeHeight - 1);
            float fz = pos.z * (d_volumeDepth  - 1);
            int ix = (int)roundf(fx);
            int iy = (int)roundf(fy);
            int iz = (int)roundf(fz);

            // Sample
            float density = sampleVolumeNearest(volumeData, d_volumeWidth, d_volumeHeight, d_volumeDepth, ix, iy, iz);

            // Basic transfer function. TODO: Move to a separate file, and then improve
            float alphaSample = density * 0.1f;
            // float alphaSample = 1.0f - expf(-density * 0.1f);
            Color3 baseColor = Color3(density, 0.1f*density, 1.f - density);  // TODO: Implement a proper transfer function

            // If density ~ 0, skip shading
            if (density > minAllowedDensity) {
                Vec3 grad = computeGradient(volumeData, d_volumeWidth, d_volumeHeight, d_volumeDepth, ix, iy, iz);
                Vec3 normal = -grad.normalize();

                Vec3 lightDir = (lightPos - pos).normalize();
                Vec3 viewDir  = -rayDir.normalize();

                // Apply Phong
                Vec3 shadedColor = phongShading(normal, lightDir, viewDir, baseColor);

                // Compose
                colorR     += (1.0f - alphaAccum) * shadedColor.x * alphaSample;
                colorG     += (1.0f - alphaAccum) * shadedColor.y * alphaSample;
                colorB     += (1.0f - alphaAccum) * shadedColor.z * alphaSample;
                alphaAccum += (1.0f - alphaAccum) * alphaSample;
            }

            tCurrent += stepSize;
        }

        accumR += colorR;
        accumG += colorG;
        accumB += colorB;
    }

    // Average samples
    accumR /= (float)SAMPLES_PER_PIXEL;
    accumG /= (float)SAMPLES_PER_PIXEL;
    accumB /= (float)SAMPLES_PER_PIXEL;

    // Final colour
    int fbIndex = (py * IMAGE_WIDTH + px) * 3;
    framebuffer[fbIndex + 0] = (unsigned char)(fminf(accumR, 1.f) * 255);
    framebuffer[fbIndex + 1] = (unsigned char)(fminf(accumG, 1.f) * 255);
    framebuffer[fbIndex + 2] = (unsigned char)(fminf(accumB, 1.f) * 255);
}

#endif // RAYCASTER_H