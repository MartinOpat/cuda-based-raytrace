#ifndef CONSTS_H
#define CONSTS_H

#include "linalg/vec.h"

// --------------------------- Basic Constants ---------------------------
const int VOLUME_WIDTH  = 49;
const int VOLUME_HEIGHT = 51;
const int VOLUME_DEPTH  = 42;

const int IMAGE_WIDTH   = 2560;
const int IMAGE_HEIGHT  = 1440;

const double epsilon = 1e-10f;
const double infty   = 1e15f;  // This vlalue is used to represent missing values in data


// --------------------------- Raycasting Constants ---------------------------
const int SAMPLES_PER_PIXEL = 8;  // TODO: Right now uses simple variance, consider using something more advanced (e.g., some commonly-used noise map)

const float alphaAcumLimit = 0.65f;   // TODO: Idk what a good accumulation value is
const float minAllowedDensity = 0.001f;

float stepSize = 0.002f;


// --------------------------- Illumination Constants ---------------------------
const double ambientStrength  = 0.3;
const double diffuseStrength  = 0.8;
const double specularStrength = 0.5;
const int shininess           = 32;

// Camera and Light
Point3 cameraPos(-0.7, -1.0, -2.0);
Vec3 cameraDir = Vec3(0.4, 0.6, 1.0).normalize();
Vec3 cameraUp = Vec3(0.0, 1.0, 0.0).normalize();
float fov = 60.0f * (M_PI / 180.0f);
Point3 lightPos(1.5, 2.0, -1.0);

#endif // CONSTS_H