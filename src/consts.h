#ifndef CONSTS_H
#define CONSTS_H

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


// --------------------------- Illumination Constants ---------------------------
const double ambientStrength  = 0.3;
const double diffuseStrength  = 0.8;
const double specularStrength = 0.5;
const int shininess           = 32;

#endif // CONSTS_H