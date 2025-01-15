#include "transferFunction.h"
#include "consts.h"

#include <stdio.h>



__device__ float opacityFromGradient(const Vec3 &grad) {
    float gradMag = grad.length();
    float alpha = 1.0f - expf(-d_opacityK * gradMag);
    return alpha;
}

__device__ float opacitySigmoid(float val) {
    return 1.0f / (1.0f + expf(d_sigmoidTwo * (val - d_sigmoidOne)));
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

  // TODO: This is a Gui select element
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
  Vec3 shadedColor = phongShading(normal, lightDir, viewDir, baseColor);  // TODO: Fix pixelated

  // Compose
  float4 result;
  result.x = shadedColor.x * alphaSample;
  result.y = shadedColor.y * alphaSample;
  result.z = shadedColor.z * alphaSample;
  result.w = alpha;

  // --------------------------- Silhouettes ---------------------------
  Vec3 N = grad.normalize();
  if (grad.length() > 0.2f && fabs(N.dot(viewDir)) < 0.02f) {
    result.x = 0.0f;
    result.y = 0.0f;
    result.z = 0.0f;
    result.w = 1.0f;
  }

  return result;
}
