#ifndef CONSTS_H
#define CONSTS_H

#include "linalg/vec.h"
#include <cmath>

// --------------------------- Basic Constants ---------------------------
// TODO: Right now this corresponds to the data set resolution (i.e., voxel per block in volume), however, we can separate this to allow for higher-resolution rendering
// const int VOLUME_WIDTH  = 576;  // lon
const int VOLUME_WIDTH  = 97;  // lon
// const int VOLUME_HEIGHT = 361;  // lat
const int VOLUME_HEIGHT = 71;  // lat
const int VOLUME_DEPTH  = 42;  // lev

const int IMAGE_WIDTH   = 1600;
const int IMAGE_HEIGHT  = 1200;

const double epsilon = 1e-10f;
const double infty   = 1e15f;  // This value is used to represent missing values in data

// --------------------------- Dataset Constants ---------------------------
const float MIN_TEMP = 210.0f;
const float MAX_TEMP = 240.0f;

const float MIN_SPEED = 0.0F;
const float MAX_SPEED = 14.0f;


// --------------------------- Raycasting Constants ---------------------------
const int SAMPLES_PER_PIXEL = 1;  // TODO: Right now uses simple variance, consider using something more advanced (e.g., some commonly-used noise map)

const float alphaAcumLimit = 1.0f;   // TODO: Idk what a good accumulation value is  <--- This finally does something when using alpha in both places at least
const float minAllowedDensity = 0.001f;

const float stepSize = 0.02f;


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
