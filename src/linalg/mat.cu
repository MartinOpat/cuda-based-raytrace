#include "mat.h"
#include <vector>
#include <algorithm>

using namespace std;

__device__ Vec3 computeGradient(float* volumeData, const int volW, const int volH, const int volD, int x, int y, int z) {
    // Finite difference for partial derivatives.
    // For boundary voxels - clamp to the boundary. 
    // Normal should point from higher to lower intensities

    int xm = max(x - 1, 0);
    int xp = min(x + 1, volW  - 1);
    int ym = max(y - 1, 0);
    int yp = min(y + 1, volH - 1);
    int zm = max(z - 1, 0);
    int zp = min(z + 1, volD  - 1);

    // Note: Assuming data is linearized (idx = z*w*h + y*w + x) TODO: Unlinearize if data not linear
    float gx = volumeData[z  * volW * volH + y  * volW + xp]
             - volumeData[z  * volW * volH + y  * volW + xm];
    float gy = volumeData[z  * volW * volH + yp * volW + x ]
             - volumeData[z  * volW * volH + ym * volW + x ];
    float gz = volumeData[zp * volW * volH + y  * volW + x ]
             - volumeData[zm * volW * volH + y  * volW + x ];

    return Vec3::init(gx, gy, gz);
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