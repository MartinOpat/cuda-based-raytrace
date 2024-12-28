// #include <iostream>
// #include <fstream>
// #include <cmath>
// #include <cuda_runtime.h>

// #include "linalg/linalg.h" 
// #include "objs/sphere.h"
// #include "img/handler.h"

// // TODO: Eventually, export this into a better place (i.e., such that we do not have to recompile every time we change a parameter)
// static const int VOLUME_WIDTH  = 1024;
// static const int VOLUME_HEIGHT = 1024;
// static const int VOLUME_DEPTH  = 1024;

// static const int IMAGE_WIDTH   = 2560;
// static const int IMAGE_HEIGHT  = 1440;

// static const int SAMPLES_PER_PIXEL = 8;  // TODO: Right now uses simple variance, consider using something more advanced (e.g., some commonly-used noise map)


// __constant__ int d_volumeWidth;
// __constant__ int d_volumeHeight;
// __constant__ int d_volumeDepth;

// static float* d_volume = nullptr;  // TODO: Adjust according to how data is loaded

// // ----------------------------------------------------------------------------------------------------
// __device__ Vec3 phongShading(const Vec3& normal, const Vec3& lightDir, const Vec3& viewDir, const Vec3& baseColor) {
//     double ambientStrength  = 0.3;
//     double diffuseStrength  = 0.8;
//     double specularStrength = 0.5;
//     int    shininess        = 32;

//     Vec3 ambient = baseColor * ambientStrength;
//     double diff = fmax(normal.dot(lightDir), 0.0);
//     Vec3 diffuse = baseColor * (diffuseStrength * diff);

//     Vec3 reflectDir = (normal * (2.0 * normal.dot(lightDir)) - lightDir).normalize();
//     double spec = pow(fmax(viewDir.dot(reflectDir), 0.0), shininess);
//     Vec3 specular = Vec3(1.0, 1.0, 1.0) * (specularStrength * spec);

//     return ambient + diffuse + specular;
// }

// // Raycast + phong
// __global__ void raycastKernel(float*  volumeData, unsigned char* framebuffer, int imageWidth, int imageHeight, Vec3 cameraPos, Vec3 cameraDir, Vec3 cameraUp, float fov, float stepSize, Vec3 lightPos) {
//     int px = blockIdx.x * blockDim.x + threadIdx.x;
//     int py = blockIdx.y * blockDim.y + threadIdx.y;
//     if (px >= imageWidth || py >= imageHeight) return;

//     float accumR = 0.0f;
//     float accumG = 0.0f;
//     float accumB = 0.0f;

//     // Multiple samples per pixel
//     for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
//         // Map to [-1, 1]
//         float u = ((px + 0.5f) / imageWidth ) * 2.0f - 1.0f;
//         float v = ((py + 0.5f) / imageHeight) * 2.0f - 1.0f;

//         // TODO: Move this (and all similar transformation code) to its own separate file
//         float tanHalfFov = tanf(fov * 0.5f);
//         u *= tanHalfFov;
//         v *= tanHalfFov;

//         // Find ray direction
//         Vec3 cameraRight = (cameraDir.cross(cameraUp)).normalize();
//         cameraUp = (cameraRight.cross(cameraDir)).normalize();
//         Vec3 rayDir = (cameraDir + cameraRight*u + cameraUp*v).normalize();

//         // Intersect (for simplicity just a 3D box from 0 to 1 in all dimensions) - TODO: Think about whether this is the best way to do this
//         float tNear = 0.f;   // TODO: These are also linear transforms, so move away
//         float tFar  = 1e6f;

//         auto intersectAxis = [&](float start, float dirVal) {
//             if (fabsf(dirVal) < 1e-10f) {  // TDDO: Add a constant - epsilon
//                 if (start < 0.f || start > 1.f) {
//                     tNear = 1e9f;
//                     tFar  = -1e9f;
//                 }
//             } else {
//                 float t0 = (0.0f - start) / dirVal;  // TODO: 0.0 and 1.0 depend on the box size -> move to a constant
//                 float t1 = (1.0f - start) / dirVal;
//                 if (t0>t1) { 
//                     float tmp=t0; 
//                     t0=t1; 
//                     t1=tmp; 
//                 }
//                 if (t0>tNear) tNear = t0;
//                 if (t1<tFar ) tFar  = t1;
//             }
//         };

//         intersectAxis(cameraPos.x, rayDir.x);
//         intersectAxis(cameraPos.y, rayDir.y);
//         intersectAxis(cameraPos.z, rayDir.z);

//         if (tNear > tFar) continue;  // No intersectionn
//         if (tNear < 0.0f) tNear = 0.0f;

//         float colorR = 0.f, colorG = 0.f, colorB = 0.f;
//         float alphaAccum = 0.f;

//         float tCurrent = tNear;
//         while (tCurrent < tFar && alphaAccum < 0.99f) {
//             Vec3 pos = cameraPos + rayDir * tCurrent;

//             // Convert to volume indices
//             float fx = pos.x * (d_volumeWidth  - 1);
//             float fy = pos.y * (d_volumeHeight - 1);
//             float fz = pos.z * (d_volumeDepth  - 1);
//             int ix = (int)roundf(fx);
//             int iy = (int)roundf(fy);
//             int iz = (int)roundf(fz);

//             // Sample
//             float density = sampleVolumeNearest(volumeData, d_volumeWidth, d_volumeHeight, d_volumeDepth, ix, iy, iz);

//             // Basic transfer function. TODO: Move to a separate file, and then improve
//             float alphaSample = density * 0.05f;
//             Vec3 baseColor = Vec3(density, 0.1f*density, 1.f - density);

//             // If density ~ 0, skip shading
//             if (density > 0.001f) {
//                 Vec3 grad = computeGradient(volumeData, d_volumeWidth, d_volumeHeight, d_volumeDepth, ix, iy, iz);
//                 Vec3 normal = -grad.normalize();

//                 Vec3 lightDir = (lightPos - pos).normalize();
//                 Vec3 viewDir  = -rayDir.normalize();

//                 // Apply Phong
//                 Vec3 shadedColor = phongShading(normal, lightDir, viewDir, baseColor);

//                 // Compose
//                 colorR     += (1.0f - alphaAccum) * shadedColor.x * alphaSample;
//                 colorG     += (1.0f - alphaAccum) * shadedColor.y * alphaSample;
//                 colorB     += (1.0f - alphaAccum) * shadedColor.z * alphaSample;
//                 alphaAccum += (1.0f - alphaAccum) * alphaSample;
//             }

//             tCurrent += stepSize;
//         }

//         accumR += colorR;
//         accumG += colorG;
//         accumB += colorB;
//     }

//     // Average samples
//     accumR /= (float)SAMPLES_PER_PIXEL;
//     accumG /= (float)SAMPLES_PER_PIXEL;
//     accumB /= (float)SAMPLES_PER_PIXEL;

//     // Final colour
//     int fbIndex = (py * imageWidth + px) * 3;
//     framebuffer[fbIndex + 0] = (unsigned char)(fminf(accumR, 1.f) * 255);
//     framebuffer[fbIndex + 1] = (unsigned char)(fminf(accumG, 1.f) * 255);
//     framebuffer[fbIndex + 2] = (unsigned char)(fminf(accumB, 1.f) * 255);
// }

// int main(int argc, char** argv) {
//     // Generate debug volume data
//     float* hostVolume = new float[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH];
//     generateVolume(hostVolume, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH);

//     // Allocate + copy data to GPU
//     size_t volumeSize = sizeof(float) * VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH;
//     cudaMalloc((void**)&d_volume, volumeSize);
//     cudaMemcpy(d_volume, hostVolume, volumeSize, cudaMemcpyHostToDevice);

//     int w = VOLUME_WIDTH, h = VOLUME_HEIGHT, d = VOLUME_DEPTH;
//     cudaMemcpyToSymbol(d_volumeWidth,  &w, sizeof(int));
//     cudaMemcpyToSymbol(d_volumeHeight, &h, sizeof(int));
//     cudaMemcpyToSymbol(d_volumeDepth,  &d, sizeof(int));

//     // Allocate framebuffer
//     unsigned char* d_framebuffer;
//     size_t fbSize = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(unsigned char);
//     cudaMalloc((void**)&d_framebuffer, fbSize);
//     cudaMemset(d_framebuffer, 0, fbSize);

//     // Camera and Light
//     Vec3 cameraPos(0.5, 0.5, -2.0);
//     Vec3 cameraDir(0.0, 0.0, 1.0);
//     Vec3 cameraUp(0.0, 1.0, 0.0);
//     float fov = 60.0f * (M_PI / 180.0f);
//     float stepSize = 0.002f;
//     Vec3 lightPos(1.5, 2.0, -1.0);

//     // Launch kernel
//     dim3 blockSize(16, 16);
//     dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1)/blockSize.x,
//                   (IMAGE_HEIGHT + blockSize.y - 1)/blockSize.y);

//     raycastKernel<<<gridSize, blockSize>>>(
//         d_volume,
//         d_framebuffer,
//         IMAGE_WIDTH,
//         IMAGE_HEIGHT,
//         cameraPos,
//         cameraDir.normalize(),
//         cameraUp.normalize(),
//         fov,
//         stepSize,
//         lightPos
//     );
//     cudaDeviceSynchronize();

//     // Copy framebuffer back to CPU
//     unsigned char* hostFramebuffer = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
//     cudaMemcpy(hostFramebuffer, d_framebuffer, fbSize, cudaMemcpyDeviceToHost);

//     // Export image
//     saveImage("output.ppm", hostFramebuffer, IMAGE_WIDTH, IMAGE_HEIGHT);

//     // Cleanup
//     delete[] hostVolume;
//     delete[] hostFramebuffer;
//     cudaFree(d_volume);
//     cudaFree(d_framebuffer);

//     std::cout << "Phong-DVR rendering done. Image saved to output.ppm" << std::endl;
//     return 0;
// }

// gpu-buffer-handler branch main
#include "hurricanedata/fielddata.h"
#include "hurricanedata/gpubufferhandler.h"
#include "hurricanedata/datareader.h"
#include "hurricanedata/gpubuffer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <iomanip> 

__global__ void middleOfTwoValues(float *ans, const FieldMetadata &fmd, FieldData fd) {
    float xi = getVal(fmd, fd, 0, 20, 100, 100);
    float yi = getVal(fmd, fd, 1, 20, 100, 100);
    *ans = (xi+yi)/2;
}

int main() {
    std::string path = "data/atmosphere_MERRA-wind-speed[179253532]";

    std::string variable = "T";

    DataReader dataReader{path, variable};

    std::cout << "created datareader\n";

    GPUBuffer buffer (dataReader);

    std::cout << "created buffer\n";

    GPUBufferHandler bufferHandler(buffer);

    float *ptr_test_read;
    cudaMallocManaged(&ptr_test_read, sizeof(float));

    std::cout << "created buffer handler\n";
    for (int i = 0; i < 10; i++) {
        FieldData fd = bufferHandler.nextFieldData();

        middleOfTwoValues<<<1, 1>>>(ptr_test_read, *bufferHandler.fmd, fd);

        cudaDeviceSynchronize();
        std::cout << "ptr_test_read = " << std::fixed << std::setprecision(6) << *ptr_test_read << "\n";
    }
    
    // TODO: measure data transfer time in this example code.
    cudaFree(ptr_test_read);
    return 0;
}