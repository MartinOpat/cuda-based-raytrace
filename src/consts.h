#ifndef CONSTS_H
#define CONSTS_H

#include "linalg/vec.h"
#include <cmath>

// --------------------------- Basic Constants ---------------------------
const int INITIAL_WINDOW_WIDTH   = 1200;
const int INITIAL_WINDOW_HEIGHT  = 900;

const double epsilon = 1e-10f;
const double infty   = 1e15f;  // This value is used to represent missing values in data

// --------------------------- Dataset Constants ---------------------------
// const int VOLUME_WIDTH  = 576;  // lon
const int VOLUME_WIDTH  = 97;  // lon
// const int VOLUME_HEIGHT = 361;  // lat
const int VOLUME_HEIGHT = 71;  // lat
const int VOLUME_DEPTH  = 42;  // lev

const float DLON = 60.0f / VOLUME_WIDTH;  // 60 for current trimmed data set range
const float DLAT = 35.0f / VOLUME_HEIGHT;  // 35 for current trimmed data set range
const float DLEV = 1000.0f / VOLUME_DEPTH;  // 1000 from max pressure (hPa) but not sure here

const float MIN_TEMP = 210.0f;
const float MAX_TEMP = 293.0f;

const float MIN_SPEED = 0.0F;
const float MAX_SPEED = 14.0f;


// --------------------------- Raycasting Constants ---------------------------
const int SAMPLES_PER_PIXEL = 4;

const float alphaAcumLimit = 0.4f;   // TODO: Atm, this only works with sigmoid
const float minAllowedDensity = 0.001f;

const float stepSize = 0.02f;


// --------------------------- Illumination Constants ---------------------------
// Shading consts
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

// Background color
extern __device__ Color3 d_backgroundColor;


// --------------------------- Transfer Function Constants ---------------------------
struct ColorStop {
    float pos;       // in [0,1]  
    Color3 color;
};

// factor for the gradient opacity function
extern __device__ float d_opacityK;
// sigmoid function variables
extern __device__ float d_sigmoidShift;
extern __device__ float d_sigmoidExp;
// combo box index
extern __device__ int d_tfComboSelected;
// constant opacity option
extern __device__ float d_opacityConst;

const int lenStopsPythonLike = 5;
const int lenStopsGrayscale = 2;
const int lenStopsBluePurpleRed = 3;
extern __constant__ ColorStop d_stopsPythonLike[5];
extern __constant__ ColorStop d_stopsGrayscale[2];
extern __constant__ ColorStop d_stopsBluePurleRed[3];

// --------------------------- Functions for handling external constants ---------------------------
void copyConstantsToDevice();

#endif // CONSTS_H
