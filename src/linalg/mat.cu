#include "mat.h"
#include "consts.h"

#include <vector>
#include <algorithm>

using namespace std;

// Samples the voxel nearest to the given coordinates.
__device__ float sampleVolumeNearest(float* volumeData, const int volW, const int volH, const int volD, int vx, int vy, int vz) {
    // For boundary voxels - clamp to the boundary. 
    if (vx < 0) vx = 0;
    if (vy < 0) vy = 0;
    if (vz < 0) vz = 0;
    if (vx >= volH) vx = volH  - 1;
    if (vy >= volW) vy = volW - 1;
    if (vz >= volD) vz = volD  - 1;

    // x <-> height, y <-> width, z <-> depth
    int idx = vz * volW * volH + vx * volW + vy;
    return volumeData[idx];
}

// tri-linear interpolation - ready if necessary (but no visible improvement for full volume)
__device__ float sampleVolumeTrilinear(float* volumeData, const int volW, const int volH, const int volD, float fx, float fy, float fz) {
    int ix = (int)floorf(fx);
    int iy = (int)floorf(fy);
    int iz = (int)floorf(fz);

    // Clamp indices to valid range
    int ix1 = min(ix + 1, volH - 1);
    int iy1 = min(iy + 1, volW - 1);
    int iz1 = min(iz + 1, volD - 1);
    ix = max(ix, 0);
    iy = max(iy, 0);
    iz = max(iz, 0);

    // Compute weights
    float dx = fx - ix;
    float dy = fy - iy;
    float dz = fz - iz;

    // Sample values
    float c00 = sampleVolumeNearest(volumeData, volW, volH, volD, ix, iy, iz) * (1.0f - dx) +
                sampleVolumeNearest(volumeData, volW, volH, volD, ix1, iy, iz) * dx;
    float c10 = sampleVolumeNearest(volumeData, volW, volH, volD, ix, iy1, iz) * (1.0f - dx) +
                sampleVolumeNearest(volumeData, volW, volH, volD, ix1, iy1, iz) * dx;
    float c01 = sampleVolumeNearest(volumeData, volW, volH, volD, ix, iy, iz1) * (1.0f - dx) +
                sampleVolumeNearest(volumeData, volW, volH, volD, ix1, iy, iz1) * dx;
    float c11 = sampleVolumeNearest(volumeData, volW, volH, volD, ix, iy1, iz1) * (1.0f - dx) +
                sampleVolumeNearest(volumeData, volW, volH, volD, ix1, iy1, iz1) * dx;

    float c0 = c00 * (1.0f - dy) + c10 * dy;
    float c1 = c01 * (1.0f - dy) + c11 * dy;

    return c0 * (1.0f - dz) + c1 * dz;
}

__device__ Vec3 computeGradient(float* volumeData, const int volW, const int volH, const int volD, float fx, float fy, float fz) {
    // Compute gradient using central differencing with trilinear interpolation
    float hx = DLAT;  // x => height => lat
    float hy = DLON;  // y => width => lon
    float hz = DLEV;  // z => depth => alt
    
    // Default
    float dfdx = (sampleVolumeTrilinear(volumeData, volW, volH, volD, fx + hx, fy, fz) -
                  sampleVolumeTrilinear(volumeData, volW, volH, volD, fx - hx, fy, fz)) / (2.0f * hx);

    float dfdy = (sampleVolumeTrilinear(volumeData, volW, volH, volD, fx, fy + hy, fz) -
                  sampleVolumeTrilinear(volumeData, volW, volH, volD, fx, fy - hy, fz)) / (2.0f * hy);

    float dfdz = (sampleVolumeTrilinear(volumeData, volW, volH, volD, fx, fy, fz + hz) -
                  sampleVolumeTrilinear(volumeData, volW, volH, volD, fx, fy, fz - hz)) / (2.0f * hz);

    // // DEBUG (TODO: Delete) - Back to nearest
    // float dfdx = (sampleVolumeNearest(volumeData, volW, volH, volD, (int)roundf(fx + 1), (int)roundf(fy), (int)roundf(fz)) -
    //               sampleVolumeNearest(volumeData, volW, volH, volD, (int)roundf(fx - 1), (int)roundf(fy), (int)roundf(fz))) / (2.0f * hx);
    // float dfdy = (sampleVolumeNearest(volumeData, volW, volH, volD, (int)roundf(fx), (int)roundf(fy + 1), (int)roundf(fz)) -
    //               sampleVolumeNearest(volumeData, volW, volH, volD, (int)roundf(fx), (int)roundf(fy - 1), (int)roundf(fz))) / (2.0f * hy);
    // float dfdz = (sampleVolumeNearest(volumeData, volW, volH, volD, (int)roundf(fx), (int)roundf(fy), (int)roundf(fz + 1))  -
    //               sampleVolumeNearest(volumeData, volW, volH, volD, (int)roundf(fx), (int)roundf(fy), (int)roundf(fz - 1))) / (2.0f * hz);

    return Vec3::init(dfdx, dfdy, dfdz);
};

// TESTING: haven't tested this function at all tbh
__device__ unsigned int packUnorm4x8(float r, float g, float b, float a) {
  union {
	  unsigned char in[4];
	  uint out;
	} u;

  float len = sqrtf(r*r + g*g + b*b + a*a);

  // This is a Vec4 but i can't be bothered to make that its own struct/class; FIXME: maybe do that if we need to? From Martin: We could use a Vec4 for rgba too, but I don't feel like it either
  u.in[0] = round(r/len * 255.0f);
  u.in[1] = round(g/len * 255.0f);
  u.in[2] = round(b/len * 255.0f);
  u.in[3] = round(a/len * 255.0f);

	return u.out;
}

// Clamp a value between a min and max value
__device__ float clamp(float value, float min, float max) {
  return fmaxf(min, fminf(value, max));
}

// Normalize a float to the range [0, 1]
__device__ float normalize(float value, float min, float max) {
  return (value - min) / (max - min);
}