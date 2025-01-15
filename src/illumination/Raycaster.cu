#include "Raycaster.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "linalg/linalg.h"
#include "consts.h"
#include "transferFunction.h"
#include "cuda_error.h"

#include <iostream>
#include <curand_kernel.h>

// TODO: instead of IMAGEWIDTH and IMAGEHEIGHT this should reflect the windowSize;
__global__ void raycastKernel(float* volumeData, FrameBuffer framebuffer, const int width, const int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;
    float accumA = 1.0f * (float)d_samplesPerPixel;

    // Initialize random state for ray scattering
    curandState randState;
    curand_init(1234, px + py * width, 0, &randState);

    // Multiple samples per pixel
    for (int s = 0; s < d_samplesPerPixel; s++) {
        // Map to [-1, 1]
        float jitterU = (curand_uniform(&randState) - 0.5f) / width;
        float jitterV = (curand_uniform(&randState) - 0.5f) / height;
        float u = ((px + 0.5f + jitterU) / width ) * 2.0f - 1.0f;
        float v = ((py + 0.5f + jitterV) / height) * 2.0f - 1.0f;

        float tanHalfFov = tanf(fov * 0.5f);
        u *= tanHalfFov;
        v *= tanHalfFov;

        // Find ray direction
        Vec3 cameraRight = (d_cameraDir.cross(d_cameraUp)).normalize();
        d_cameraUp = (cameraRight.cross(d_cameraDir)).normalize();
        Vec3 rayDir = (d_cameraDir + cameraRight*u + d_cameraUp*v).normalize();

        // Intersect
        float tNear = 0.0f;
        float tFar  = 1e6f;
        auto intersectAxis = [&](float start, float dir, float minV, float maxV) {
            if (fabsf(dir) < epsilon) {
                // Ray parallel to axis. If outside min..max, no intersection.
                if (start < minV || start > maxV) {
                    tNear = 1e9f;
                    tFar  = -1e9f;
                }
            } else {
                float t0 = (minV - start) / dir;
                float t1 = (maxV - start) / dir;
                if (t0 > t1) {
                    float tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                }
                if (t0 > tNear) tNear = t0;
                if (t1 < tFar ) tFar  = t1;
            }
        };

        intersectAxis(d_cameraPos.x, rayDir.x, 0.0f, (float)VOLUME_HEIGHT);
        intersectAxis(d_cameraPos.y, rayDir.y, 0.0f, (float)VOLUME_WIDTH);
        intersectAxis(d_cameraPos.z, rayDir.z, 0.0f, (float)VOLUME_DEPTH);

        if (tNear > tFar) {
          // No intersection -> Set to brackground color (multiply by d_samplesPerPixel because we divide by it later)
          accumR = d_backgroundColor.x * (float)d_samplesPerPixel;
          accumG = d_backgroundColor.y * (float)d_samplesPerPixel;
          accumB = d_backgroundColor.z * (float)d_samplesPerPixel;
          accumA = 1.0f * (float)d_samplesPerPixel;
          
        } else {
          if (tNear < 0.0f) tNear = 0.0f;

          float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f;
          float alphaAccum = 0.0f;

          float t = tNear;  // Front to back
          while (t < tFar && alphaAccum < d_alphaAcumLimit) {
              Point3 pos = d_cameraPos + rayDir * t;

              // Convert to volume indices
              int ix = (int)roundf(pos.x);
              int iy = (int)roundf(pos.y);
              int iz = (int)roundf(pos.z);

              // Sample (pick appropriate method based on volume size) TODO: Consider adding a way to pick this in GUI (?)
              // float density = sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, ix, iy, iz);
              float density = sampleVolumeTrilinear(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, pos.x, pos.y, pos.z);

              // If density ~ 0, skip shading
              if (density > minAllowedDensity) {
                Vec3 grad = computeGradient(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, pos.x, pos.y, pos.z);
                float4 color = transferFunction(density, grad, pos, rayDir);  // This already returns the alpha-weighted color

                //Accumulate color, and alpha
                colorR = (1.0f - alphaAccum) * color.x + colorR;
                colorG = (1.0f - alphaAccum) * color.y + colorG;
                colorB = (1.0f - alphaAccum) * color.z + colorB;
                alphaAccum = (1 - alphaAccum) * color.w + alphaAccum;

              }


              t += stepSize;
          }


          // Calculate final colour
          accumR += colorR;
          accumG += colorG;
          accumB += colorB;
          accumA += alphaAccum;

          // Blend with background (for transparency)
          float leftover = 1.0 - alphaAccum;
          accumR = accumR + leftover * d_backgroundColor.x;
          accumG = accumG + leftover * d_backgroundColor.y;
          accumB = accumB + leftover * d_backgroundColor.z;
        }
    }


    // Average samples
    accumR /= (float)d_samplesPerPixel;
    accumG /= (float)d_samplesPerPixel;
    accumB /= (float)d_samplesPerPixel;
    accumA /= (float)d_samplesPerPixel;

    // Final colour
    framebuffer.writePixel(px, py, accumR, accumG, accumB, accumA);
}


Raycaster::Raycaster(cudaGraphicsResource_t resources, int w, int h, float* data) {
	this->resources = resources;
	this->w = w;
	this->h = h;

	this->fb = new FrameBuffer(w, h);
  this->data = data;

	// camera_info = CameraInfo(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), 90.0f, (float) w, (float) h);
	// d_camera = thrust::device_new<Camera*>();

	check_cuda_errors(cudaDeviceSynchronize());
}


void Raycaster::render() {
  check_cuda_errors(cudaGraphicsMapResources(1, &this->resources));
	check_cuda_errors(cudaGraphicsResourceGetMappedPointer((void**)&(this->fb->buffer), &(this->fb->buffer_size), resources));

  // FIXME: might not be the best parallelization configuration
	int tx = 8;
	int ty = 8;
	dim3 threadSize(this->w / tx + 1, this->h / ty + 1);
	dim3 blockSize(tx, ty);

  // TODO: pass camera info at some point
	// frame buffer is implicitly copied to the device each frame
  raycastKernel<<<threadSize, blockSize>>> (this->data, *this->fb, this->w, this->h);

  check_cuda_errors(cudaGetLastError());
  check_cuda_errors(cudaDeviceSynchronize());
  check_cuda_errors(cudaGraphicsUnmapResources(1, &this->resources));
}


void Raycaster::resize(int w, int h) {
  this->w = w;
  this->h = h;

  delete this->fb;  
  this->fb = new FrameBuffer(w, h);

  // TODO: should be globals probably
	int tx = 8;
	int ty = 8;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);

  check_cuda_errors(cudaDeviceSynchronize());
}
