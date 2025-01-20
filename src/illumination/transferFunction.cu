#include "transferFunction.h"
#include "consts.h"

#include <stdio.h>

// [Levoy 1988]
__device__ float levoyOpacity(const Vec3 &grad, float val) {
    float r = d_levoyWidth; // width
    float fv = d_levoyFocus; // chosen value
    float epsilon = 1E-8;
    float gradMag = grad.length();
    if ((gradMag < epsilon) && ((val - epsilon) <= fv) && (fv <= (val + epsilon))) return 1.0f;
    if (gradMag < epsilon) return 0.0f;
    float lowBound = val - r*gradMag;
    float upperBound = val + r*gradMag;
    // float lowBound = fv - r;
    // float upperBound = fv + r;
    if (!((lowBound <= fv) && (fv <= upperBound))) return 0.0f;
    // if ((lowBound <= gradMag) && (gradMag <= upperBound)) return 1.0f;
    // return 0.0f;

    float alpha = d_opacityConst*(1 - (1/r)*fabs((fv-val)/gradMag));
    return alpha;
}

__device__ float opacityFromGradient(const Vec3 &grad, const Vec3& rayDir) {
    float gradMag = grad.length();
    // float gradMag = 1-fabs(grad.normalize().dot(rayDir));
    // float gradMag = grad.length()*(1-fabs(grad.normalize().dot(rayDir)));  // Alternative, but not particularly better
    float alpha = 1.0f - expf(-d_opacityK * gradMag);
    return alpha;
}

__device__ float opacitySigmoid(float val) {
    return 1.0f / (1.0f + expf(d_sigmoidExp * (val - d_sigmoidShift)));
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
  // TODO: Somehow pick if to use temp of speed normalization ... or pass extremas as params. <-If we decide to visualize more than 1 type of data
  // float normDensity = (density - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
  float normDensity = (density - 273) / (MAX_TEMP - MIN_TEMP)+16.f/21.f;  // Make zero match Celsius zero

  // float normDensity = (density - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);

  normDensity = clamp(normDensity, 0.0f, 1.0f);

  // --------------------------- Map density to color ---------------------------
  // Pick color map
  Color3 baseColor;
  switch (d_tfComboSelectedColor) {
  case 0:
    baseColor = colorMap(normDensity, d_stopsPythonLike, lenStopsPythonLike);
    break;
  
  case 1:
    baseColor = colorMap(normDensity, d_stopsBluePurleRed, lenStopsBluePurpleRed);
    break;

  case 2:
    baseColor = colorMap(normDensity, d_stopsGrayscale, lenStopsGrayscale);
    break;

  default:
    baseColor = colorMap(normDensity, d_stopsPythonLike, lenStopsPythonLike);
    break;
  }

  // Pick opacity function
  float alpha;
  switch (d_tfComboSelected) {
  case 0:
    alpha = opacityFromGradient(grad, rayDir);
    break;
  
  case 1:
    alpha = opacitySigmoid(normDensity);
    break;

  case 2:
    alpha = d_opacityConst;
    break;

  case 3:
    alpha = levoyOpacity(grad, normDensity);
    break;
  
  default:
    alpha = 1.0f;  // This should not be reached anyway
    break;
  }

  float alphaSample = density * alpha * 0.1;  // TODO: Why is this still 0.1?

  // --------------------------- Shading ---------------------------
  // Apply Phong
  Vec3 normal = -grad.normalize();
  Vec3 lightDir = (d_lightPos - pos).normalize();
  Vec3 viewDir  = -rayDir.normalize();
  Vec3 shadedColor = phongShading(normal, lightDir, viewDir, baseColor);  // TODO: Check if still pixelated

  // Compose
  float4 result;
  result.x = shadedColor.x * alphaSample;
  result.y = shadedColor.y * alphaSample;
  result.z = shadedColor.z * alphaSample;
  result.w = alpha;

  // --------------------------- Silhouettes ---------------------------
  Vec3 N = grad.normalize();
  if (d_showSilhouettes && grad.length() > 0.2f && fabs(N.dot(viewDir)) < d_silhouettesThreshold) {
    result.x = 0.0f;
    result.y = 0.0f;
    result.z = 0.0f;
    result.w = alpha;
    // result.w = d_opacityConst;
  }

  return result;
}
