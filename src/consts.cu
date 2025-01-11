#include "consts.h"

// ----------------------- Colour mapping -----------------------
__constant__ ColorStop d_stopsPythonLike[5];
__constant__ ColorStop d_stopsGrayscale[2];
__constant__ ColorStop d_stopsBluePurleRed[3];

const ColorStop h_stopsPythonLike[] = {
        { 0.0f, Color3::init(0.2298057f, 0.29871797f, 0.75368315f) }, // Dark Blue
        { 0.25f, Color3::init(0.23437708f, 0.30554173f, 0.75967953f) }, // Mid Blue
        { 0.5f, Color3::init(0.27582712f, 0.36671692f, 0.81255294f) }, // White
        { 0.75f, Color3::init(0.79606387f, 0.84869321f, 0.93347147f) }, // Light Orange
        { 1.0f, Color3::init(0.70567316f, 0.01555616f, 0.15023281f) }  // Red
};

const ColorStop h_stopsGrayscale[] = {
        { 0.0f, Color3::init(0.0f, 0.0f, 0.0f) },  // No colour
        { 1.0f, Color3::init(1.0f, 1.0f, 1.0f) }   // White
};

const ColorStop h_stopsBluePurleRed[] = {
        { 0.0f, Color3::init(0.0f, 0.0f, 1.0f) },   // deep blue
        { 0.5f, Color3::init(0.5f, 0.0f, 0.5f) },   // purple
        { 1.0f, Color3::init(1.0f, 0.0f, 0.0f) }    // deep red
};

// ----------------------- Camera and Light -----------------------

__device__ Point3 d_cameraPos;
__device__ Vec3 d_cameraDir;
__device__ Vec3 d_cameraUp;
__device__ Point3 d_lightPos;

// Point3 h_cameraPos = Point3::init(300.0f, 200.0f, -700.0f);  // Camera for full data set
Point3 h_cameraPos = Point3::init(50.0f, -50.0f, -75.0f);  // Camera for partially trimmed data set (TODO: Probably upside down atm)
Vec3 center = Vec3::init((float)VOLUME_WIDTH/2.0f, (float)VOLUME_HEIGHT/2.0f, (float)VOLUME_DEPTH/2.0f);
Vec3 h_cameraDir = (center - h_cameraPos).normalize();
Vec3 h_cameraUp = Vec3::init(0.0, 0.0, 1.0).normalize();
Point3 h_lightPos = Point3::init(1.5, 2.0, -1.0);


// Copy the above values to the device
void copyConstantsToDevice() {
    // ----------------------- Colour mapping -----------------------
    cudaMemcpyToSymbol(d_stopsPythonLike, h_stopsPythonLike, sizeof(h_stopsPythonLike));
    cudaMemcpyToSymbol(d_stopsGrayscale, h_stopsGrayscale, sizeof(h_stopsGrayscale));
    cudaMemcpyToSymbol(d_stopsBluePurleRed, h_stopsBluePurleRed, sizeof(h_stopsBluePurleRed));


    // ----------------------- Camera and Light -----------------------
    cudaMemcpyToSymbol(d_cameraPos, &h_cameraPos, sizeof(Point3));
    cudaMemcpyToSymbol(d_cameraDir, &h_cameraDir, sizeof(Vec3));
    cudaMemcpyToSymbol(d_cameraUp, &h_cameraUp, sizeof(Vec3));
    cudaMemcpyToSymbol(d_lightPos, &h_lightPos, sizeof(Point3));
}
