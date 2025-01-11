#include "transferFunction.h"


// Samples the voxel nearest to the given coordinates.
__device__ float sampleVolumeNearest(float* volumeData, const int volW, const int volH, const int volD, int vx, int vy, int vz) {
    // x <-> height, y <-> width, z <-> depth <--- So far this is the best one
    if (vx < 0) vx = 0;
    if (vy < 0) vy = 0;
    if (vz < 0) vz = 0;
    if (vx >= volH) vx = volH  - 1;
    if (vy >= volW) vy = volW - 1;
    if (vz >= volD) vz = volD  - 1;

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


__device__ float opacityFromGradient(const Vec3 &grad) {
    float gradMag = grad.length();
    float k = 1e-6f;               // tweak (the smaller the value, the less opacity)  // TODO: Add a slider for this
    float alpha = 1.0f - expf(-k * gradMag);
    return alpha;
}

__device__ float opacitySigmoid(float val) {
    return 1.0f / (1.0f + expf(-250.f * (val - 0.5f)));  // TODO: Parametrize and add sliders
}

__device__ Color3 colorMap(float normalizedValues, const ColorStop stops[], int N) {
    // clamp to [0,1]
    normalizedValues = fminf(fmaxf(normalizedValues, 0.0f), 1.0f);

    // N stops => N-1 intervals
    for (int i = 0; i < N - 1; ++i) {
        float start = stops[i].pos;
        float end   = stops[i + 1].pos;

        if (normalizedValues >= start && normalizedValues <= end) {
            float localT = (normalizedValues - start) / (end - start);
            return interpolate(stops[i].color, stops[i + 1].color, localT);
        }
    }

    // fallback if something goes out of [0,1] or numerical issues
    return stops[N - 1].color;
}


// Transfer function
__device__ float4 transferFunction(float density, const Vec3& grad, const Point3& pos, const Vec3& rayDir) {
  
  // --------------------------- Sample the volume ---------------------------
  // TODO: Somehow pick if to use temp of speed normalization ... or pass extremas as params.
  float normDensity = (density - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
  // float normDensity = (density - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);

  normDensity = clamp(normDensity, 0.0f, 1.0f);

  // --------------------------- Map density to color ---------------------------
  // TODO: Add a way to pick stops here
  Color3 baseColor = colorMap(normDensity, d_stopsPythonLike, lenStopsPythonLike);

  // TODO: Add a way to pick different function for alpha
  float alpha = opacityFromGradient(grad);
  // alpha = 0.1f;
  alpha = opacitySigmoid(normDensity);
  // alpha = (1.0f - fabs(grad.normalize().dot(rayDir.normalize()))) * 0.8f + 0.2f;

  float alphaSample = density * alpha * 0.1;

  // --------------------------- Shading ---------------------------
  // Apply Phong
  Vec3 normal = -grad.normalize();
  Vec3 lightDir = (d_lightPos - pos).normalize();
  Vec3 viewDir  = -rayDir.normalize();
  Vec3 shadedColor = phongShading(normal, lightDir, viewDir, baseColor);

  // Compose
  float4 result;
  result.x = shadedColor.x * alphaSample;
  result.y = shadedColor.y * alphaSample;
  result.z = shadedColor.z * alphaSample;
  result.w = alpha;

  // --------------------------- Silhouettes ---------------------------
  // TODO: This is the black silhouette, technically if we are doing alpha based on gradient then it's kind of redundant (?) ... but could also be used for even more pronounced edges
  // TODO: Add a way to adjust the treshold (0.2f atm)
  // TODO: I don't think we should literally be doing this => use gradient based opacity => delete the below if-statement
  // if (fabs(grad.normalize().dot(rayDir.normalize())) < 0.2f) {
  //   result.x = 0.0f;
  //   result.y = 0.0f;
  //   result.z = 0.0f;
  //   result.w = 1.0f;
  // }

  return result;
}
