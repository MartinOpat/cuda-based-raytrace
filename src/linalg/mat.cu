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

__device__ Vec3 computeGradient(float* volumeData, const int volW, const int volH, const int volD, int vx, int vy, int vz) {
    // Finite difference for partial derivatives.
    

    float dfdx = (sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, vx + 1, vy, vz) -
                  sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, vx - 1, vy, vz)) / (2.0f * DLAT);  // x => height => lat

    float dfdy = (sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, vx, vy + 1, vz) -
                  sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, vx, vy - 1, vz)) / (2.0f * DLON);  // y => width => lon

    float dfdz = (sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, vx, vy, vz + 1) -
                  sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, vx, vy, vz - 1)) / (2.0f * DLEV);

    return Vec3::init(dfdx, dfdy, dfdz);
};

// TESTING: haven't tested this function at all tbh
__device__ unsigned int packUnorm4x8(float r, float g, float b, float a) {
  union {
	  unsigned char in[4];
	  uint out;
	} u;

  float len = sqrtf(r*r + g*g + b*b + a*a);

  // This is a Vec4 but i can't be bothered to make that its own struct/class; FIXME: maybe do that if we need to?
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