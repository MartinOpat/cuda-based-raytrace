#include "consts.h"

__device__ Point3 d_cameraPos;
__device__ Vec3 d_cameraDir;
__device__ Vec3 d_cameraUp;
__device__ Point3 d_lightPos;

Point3 h_cameraPos = Point3::init(-300.0f, 200.0f, -300.0f);
Vec3 center = Vec3::init((float)VOLUME_WIDTH/2.0f, (float)VOLUME_HEIGHT/2.0f, (float)VOLUME_DEPTH/2.0f);
Vec3 h_cameraDir = (center - h_cameraPos).normalize();
Vec3 h_cameraUp = Vec3::init(0.0, 1.0, 0.0).normalize();
Point3 h_lightPos = Point3::init(1.5, 2.0, -1.0);

void copyConstantsToDevice() {
    cudaMemcpyToSymbol(d_cameraPos, &h_cameraPos, sizeof(Point3));
    cudaMemcpyToSymbol(d_cameraDir, &h_cameraDir, sizeof(Vec3));
    cudaMemcpyToSymbol(d_cameraUp, &h_cameraUp, sizeof(Vec3));
    cudaMemcpyToSymbol(d_lightPos, &h_lightPos, sizeof(Point3));
}
