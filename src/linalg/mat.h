#pragma once

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
}
