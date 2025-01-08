#ifndef CONSTS_H
#define CONSTS_H

#include "linalg/vec.h"
#include <cmath>

// --------------------------- Basic Constants ---------------------------
const int VOLUME_WIDTH  = 49;
const int VOLUME_HEIGHT = 51;
const int VOLUME_DEPTH  = 42;

const int IMAGE_WIDTH   = 800;
const int IMAGE_HEIGHT  = 600;

const double epsilon = 1e-10f;
const double infty   = 1e15f;  // This value is used to represent missing values in data


// --------------------------- Raycasting Constants ---------------------------
const int SAMPLES_PER_PIXEL = 8;  // TODO: Right now uses simple variance, consider using something more advanced (e.g., some commonly-used noise map)

const float alphaAcumLimit = 0.65f;   // TODO: Idk what a good accumulation value is
const float minAllowedDensity = 0.001f;

const float stepSize = 0.002f;


// --------------------------- Illumination Constants ---------------------------
const double ambientStrength  = 0.3;
const double diffuseStrength  = 0.8;
const double specularStrength = 0.5;
const int shininess           = 32;
const float fov             = 60.0f * (M_PI / 180.0f);

// Camera and Light
extern __device__ Point3 d_cameraPos;
extern __device__ Vec3 d_cameraDir;
extern __device__ Vec3 d_cameraUp;
extern __device__ Point3 d_lightPos;

// --------------------------- Functions for handling external constants ---------------------------
void copyConstantsToDevice();

#endif // CONSTS_H
