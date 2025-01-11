#include "Raycaster.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "linalg/linalg.h"
#include "consts.h"
#include "cuda_error.h"
#include "shading.h"
#include <iostream>
#include "objs/sphere.h"
#include <curand_kernel.h>

// TODO: Probbably move this transfer function business into a different file
// Samples the voxel nearest to the given coordinates. TODO: Can be re-used in other places so move
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
    int ix1 = min(ix + 1, volW - 1);
    int iy1 = min(iy + 1, volH - 1);
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
    float gradMag = grad.length();  // magnitude
    float k = 1e-4f;               // tweak (the smaller the value, the less opacity)  // TODO: What should be the value of this?
    float alpha = 1.0f - expf(-k * gradMag);
    return alpha;
}

struct ColorStop
{
    float pos;       // in [0,1]  
    Color3 color;    // R,G,B in [0,1]
};

// TODO: Rename probably
__device__ Color3 colorMapViridis(float normalizedT) {
    // Here we redefine the color stops to go from deep blue (0.0) to purple (0.5) to deep red (1.0)
    ColorStop tempStops[] = {
        { 0.0f, Color3::init(0.0f, 0.0f, 1.0f) },   // deep blue
        { 0.5f, Color3::init(0.5f, 0.0f, 0.5f) },   // purple
        { 1.0f, Color3::init(1.0f, 0.0f, 0.0f) }    // deep red
    };

    // Clamp to [0,1]
    if (normalizedT < 0.0f) normalizedT = 0.0f;
    if (normalizedT > 1.0f) normalizedT = 1.0f;

    // We have 3 stops => 2 intervals
    const int N = 3;
    for (int i = 0; i < N - 1; ++i)
    {
        float start = tempStops[i].pos;
        float end   = tempStops[i + 1].pos;

        if (normalizedT >= start && normalizedT <= end)
        {
            float localT = (normalizedT - start) / (end - start);
            return interpolate(tempStops[i].color, tempStops[i + 1].color, localT);
        }
    }
    // Fallback if something goes out of [0,1] or numerical issues
    return tempStops[N - 1].color;
}

// Monochromatic colormap for speed
__device__ Color3 colorMapMonochrome(float normalizedSpeed) {
    // Define the color stops: black (0.0) to white (1.0)
    ColorStop speedStops[] = {
        { 0.0f, Color3::init(0.0f, 0.0f, 0.0f) },  // No colour
        { 1.0f, Color3::init(1.0f, 1.0f, 1.0f) }   // White
    };

    // Clamp to [0,1]
    if (normalizedSpeed < 0.0f) normalizedSpeed = 0.0f;
    if (normalizedSpeed > 1.0f) normalizedSpeed = 1.0f;

    // Single interval (N = 2 for black to white)
    const int N = 2;
    for (int i = 0; i < N - 1; ++i)
    {
        float start = speedStops[i].pos;
        float end   = speedStops[i + 1].pos;

        if (normalizedSpeed >= start && normalizedSpeed <= end)
        {
            float localT = (normalizedSpeed - start) / (end - start);
            return interpolate(speedStops[i].color, speedStops[i + 1].color, localT);
        }
    }

    // Fallback
    return speedStops[N - 1].color;
}

// Colormap function for gradient from blue to white to red
__device__ Color3 colorMapPythonLike(float normalizedValue) {
  // Define control points for the colormap (approximation of coolwarm)
  ColorStop coolwarmStops[] = {
        { 0.0f, Color3::init(0.2298057f, 0.29871797f, 0.75368315f) }, // Dark Blue
        { 0.25f, Color3::init(0.23437708f, 0.30554173f, 0.75967953f) }, // Mid Blue
        { 0.5f, Color3::init(0.27582712f, 0.36671692f, 0.81255294f) }, // White
        { 0.75f, Color3::init(0.79606387f, 0.84869321f, 0.93347147f) }, // Light Orange
        { 1.0f, Color3::init(0.70567316f, 0.01555616f, 0.15023281f) }  // Red
  };

  // Clamp the scalar value to [0, 1]
  normalizedValue = fminf(fmaxf(normalizedValue, 0.0f), 1.0f);

  // Interpolate between the defined color stops
  const int N = 5; // Number of control points
  for (int i = 0; i < N - 1; ++i) {
      float start = coolwarmStops[i].pos;
      float end = coolwarmStops[i + 1].pos;

      if (normalizedValue >= start && normalizedValue <= end) {
          float localT = (normalizedValue - start) / (end - start);
          return interpolate(coolwarmStops[i].color, coolwarmStops[i + 1].color, localT);
      }
  }

  // Fallback (shouldn't reach here due to clamping)
  return coolwarmStops[N - 1].color;
}


// Transfer function
__device__ float4 transferFunction(float density, const Vec3& grad, const Point3& pos, const Vec3& rayDir) {
  // Basic transfer function. TODO: Move to a separate file, and then improve

  // Color3 baseColor = Color3::init(density, 0.1f*density, 1.f - density);  // TODO: Implement a proper transfer function
  // Color3 baseColor = temperatureToRGB(density);
  
  float normDensity = (density - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
  // float normDensity = (density - MIN_SPEED) / (MAX_SPEED - MIN_SPEED);

  normDensity = clamp(normDensity, 0.0f, 1.0f);

  // Color3 baseColor = colorMapViridis(normDensity);
  // Color3 baseColor = colorMapMonochrome(normDensity);
  Color3 baseColor = colorMapPythonLike(normDensity);


  float alpha = opacityFromGradient(grad);
  // alpha = 0.1f;
  alpha = 1.0f / (1.0f + expf(-250.f * (normDensity - 0.5f)));  // This is also quite nice, but the exponent should be parameterized
  float alphaSample = density * alpha;  // TODO: Decide whether to keep alpha here or not

  Vec3 normal = -grad.normalize();

  Vec3 lightDir = (d_lightPos - pos).normalize();
  Vec3 viewDir  = -rayDir.normalize();

  // Apply Phong
  Vec3 shadedColor = phongShading(normal, lightDir, viewDir, baseColor);

  // Compose
  float4 result;
  result.x = shadedColor.x * alphaSample;
  result.y = shadedColor.y * alphaSample;
  result.z = shadedColor.z * alphaSample;
  result.w = alpha;  // TODO: Again, decide if alpha here is correct or not

  // TODO: This is the black silhouette, technically if we are doing alpha based on gradient then it's kind of redundant (?) ... but could also be used for even sharper edges
  if (grad.length() > epsilon && fabs(grad.normalize().dot(viewDir)) < 0.2f) {
    result.x = 0.0f;
    result.y = 0.0f;
    result.z = 0.0f;
    result.w = 1.0f;
  }

  return result;
}



// TODO: instead of IMAGEWIDTH and IMAGEHEIGHT this should reflect the windowSize;
__global__ void raycastKernel(float* volumeData, FrameBuffer framebuffer) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= IMAGE_WIDTH || py >= IMAGE_HEIGHT) return;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;
    float accumA = 1.0f;
    // Initialize random state
    curandState randState;
    curand_init(1234, px + py * IMAGE_WIDTH, 0, &randState);

    // Multiple samples per pixel
    // Multiple samples per pixel
    for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
        // Map to [-1, 1]
        float jitterU = (curand_uniform(&randState) - 0.5f) / IMAGE_WIDTH;
        float jitterV = (curand_uniform(&randState) - 0.5f) / IMAGE_HEIGHT;
        float u = ((px + 0.5f + jitterU) / IMAGE_WIDTH ) * 2.0f - 1.0f;
        float v = ((py + 0.5f + jitterV) / IMAGE_HEIGHT) * 2.0f - 1.0f;

        // TODO: Move this (and all similar transformation code) to its own separate file
        float tanHalfFov = tanf(fov * 0.5f);
        u *= tanHalfFov;
        v *= tanHalfFov;

        // Find ray direction
        Vec3 cameraRight = (d_cameraDir.cross(d_cameraUp)).normalize();
        d_cameraUp = (cameraRight.cross(d_cameraDir)).normalize();
        Vec3 rayDir = (d_cameraDir + cameraRight*u + d_cameraUp*v).normalize();

        // Intersect
        float tNear = 0.0f;
        float tFar  = 1e6f;
        auto intersectAxis = [&](float start, float dir, float minV, float maxV) {
            if (fabsf(dir) < epsilon) {
                // Ray parallel to axis. If outside min..max, no intersection.
                if (start < minV || start > maxV) {
                    tNear = 1e9f;
                    tFar  = -1e9f;
                }
            } else {
                float t0 = (minV - start) / dir;
                float t1 = (maxV - start) / dir;
                if (t0 > t1) {
                    float tmp = t0;
                    t0 = t1;
                    t1 = tmp;
                }
                if (t0 > tNear) tNear = t0;
                if (t1 < tFar ) tFar  = t1;
            }
        };

        intersectAxis(d_cameraPos.x, rayDir.x, 0.0f, (float)VOLUME_HEIGHT);
        intersectAxis(d_cameraPos.y, rayDir.y, 0.0f, (float)VOLUME_WIDTH);
        intersectAxis(d_cameraPos.z, rayDir.z, 0.0f, (float)VOLUME_DEPTH);

        if (tNear > tFar) {
          // No intersection -> Set to brackground color (multiply by SAMPLES_PER_PIXEL because we divide by it later)
          accumR = 0.1f * (float)SAMPLES_PER_PIXEL;
          accumG = 0.1f * (float)SAMPLES_PER_PIXEL;
          accumB = 0.1f * (float)SAMPLES_PER_PIXEL;
          accumA = 1.0f * (float)SAMPLES_PER_PIXEL;
        } else {
          if (tNear < 0.0f) tNear = 0.0f;

          float colorR = 0.0f, colorG = 0.0f, colorB = 0.0f;
          float alphaAccum = 0.0f;

          float t = tNear;
          while (t < tFar && alphaAccum < alphaAcumLimit) {
              Point3 pos = d_cameraPos + rayDir * t;

              // Convert to volume indices
              int ix = (int)roundf(pos.x);
              int iy = (int)roundf(pos.y);
              int iz = (int)roundf(pos.z);

              // Sample (pick appropriate method based on volume size)
              float density = sampleVolumeNearest(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, ix, iy, iz);
              // float density = sampleVolumeTrilinear(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, pos.x, pos.y, pos.z);

              // If density ~ 0, skip shading
              if (density > minAllowedDensity) {
                Vec3 grad = computeGradient(volumeData, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH, ix, iy, iz);
                float4 color = transferFunction(density, grad, pos, rayDir);  // This already returns the alpha-weighted color

                // // Accumulate color, respecting current transparency
                // colorR += (1.0f - alphaAccum) * color.x * color.w;
                // colorG += (1.0f - alphaAccum) * color.y * color.w;
                // colorB += (1.0f - alphaAccum) * color.z * color.w;

                // // Update the total accumulated alpha
                // alphaAccum += (1.0f - alphaAccum) * color.w;

                colorR = (1.0f - alphaAccum) * color.x + colorR;
                colorG = (1.0f - alphaAccum) * color.y + colorG;
                colorB = (1.0f - alphaAccum) * color.z + colorB;
                alphaAccum = (1 - alphaAccum) * color.w + alphaAccum;

              }


              t += stepSize;
          }


          // Calculate final colour
          accumR += colorR;
          accumG += colorG;
          accumB += colorB;
          accumA += alphaAccum;

          // Blend with background (for transparency)  // TODO: Put background colour in a constant
          float leftover = 1.0 - alphaAccum;
          accumR = accumR + leftover * 0.1f;
          accumG = accumG + leftover * 0.1f;
          accumB = accumB + leftover * 0.1f;
          // accumA = accumA + leftover * 0.0f;  // This is set to 0.9 instead of 1.0 to be able to see where the volume is a little bit
        }
    }


    // Average samples
    accumR /= (float)SAMPLES_PER_PIXEL;
    accumG /= (float)SAMPLES_PER_PIXEL;
    accumB /= (float)SAMPLES_PER_PIXEL;
    accumA /= (float)SAMPLES_PER_PIXEL;

    // Final colour
    framebuffer.writePixel(px, py, accumR, accumG, accumB, accumA);
    // int fbIndex = (py * IMAGE_WIDTH + px) * 3;
    // framebuffer[fbIndex + 0] = (unsigned char)(fminf(accumR, 1.f) * 255);
    // framebuffer[fbIndex + 1] = (unsigned char)(fminf(accumG, 1.f) * 255);
    // framebuffer[fbIndex + 2] = (unsigned char)(fminf(accumB, 1.f) * 255);
}


Raycaster::Raycaster(cudaGraphicsResource_t resources, int w, int h, float* data) {
	this->resources = resources;
	this->w = w;
	this->h = h;

	this->fb = new FrameBuffer(w, h);
  this->data = data;

	// camera_info = CameraInfo(Vec3(0.0f, 0.0f, 0.0f), Vec3(0.0f, 0.0f, 0.0f), 90.0f, (float) w, (float) h);
	// d_camera = thrust::device_new<Camera*>();

	check_cuda_errors(cudaDeviceSynchronize());
}


void Raycaster::render() {
  check_cuda_errors(cudaGraphicsMapResources(1, &this->resources));
	check_cuda_errors(cudaGraphicsResourceGetMappedPointer((void**)&(this->fb->buffer), &(this->fb->buffer_size), resources));

  // FIXME: might not be the best parallelization configuraiton
	int tx = 16;
	int ty = 16;
	dim3 threadSize(this->w / tx + 1, this->h / ty + 1);
	dim3 blockSize(tx, ty);

  // TODO: pass camera info at some point
	// frame buffer is implicitly copied to the device each frame
  raycastKernel<<<threadSize, blockSize>>> (this->data, *this->fb);

  check_cuda_errors(cudaGetLastError());
  check_cuda_errors(cudaDeviceSynchronize());
  check_cuda_errors(cudaGraphicsUnmapResources(1, &this->resources));
}


void Raycaster::resize(int w, int h) {
  this->w = w;
  this->h = h;

  delete this->fb;  
  this->fb = new FrameBuffer(w, h);

  // TODO: should be globals probably
	int tx = 8;
	int ty = 8;

	dim3 blocks(w / tx + 1, h / ty + 1);
	dim3 threads(tx, ty);

  check_cuda_errors(cudaDeviceSynchronize());
}
