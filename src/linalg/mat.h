#pragma once

__device__ Vec3 computeGradient(float* volumeData, const int d_volumeWidth, const int d_volumeHeight, const int d_volumeDepth, int x, int y, int z) {
    // Finite difference for partial derivatives.
    // For boundary voxels - clamp to the boundary. 
    // Normal should point from higher to lower intensities

    int xm = max(x - 1, 0);
    int xp = min(x + 1, d_volumeWidth  - 1);
    int ym = max(y - 1, 0);
    int yp = min(y + 1, d_volumeHeight - 1);
    int zm = max(z - 1, 0);
    int zp = min(z + 1, d_volumeDepth  - 1);

    // Note: Assuming data is linearized (idx = z*w*h + y*w + x) TODO: Unlinearize if data not linear
    float gx = volumeData[z  * d_volumeWidth * d_volumeHeight + y  * d_volumeWidth + xp]
             - volumeData[z  * d_volumeWidth * d_volumeHeight + y  * d_volumeWidth + xm];
    float gy = volumeData[z  * d_volumeWidth * d_volumeHeight + yp * d_volumeWidth + x ]
             - volumeData[z  * d_volumeWidth * d_volumeHeight + ym * d_volumeWidth + x ];
    float gz = volumeData[zp * d_volumeWidth * d_volumeHeight + y  * d_volumeWidth + x ]
             - volumeData[zm * d_volumeWidth * d_volumeHeight + y  * d_volumeWidth + x ];

    return Vec3(gx, gy, gz);
}
