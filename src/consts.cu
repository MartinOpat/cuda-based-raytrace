#include "consts.h"
#include "cuda_error.h"

// ----------------------- Colour mapping -----------------------
// __device__ __constant__ ColorStop d_stopsPythonLike[5];
__device__ __constant__ ColorStop d_stopsPythonLike[11];
__device__ __constant__ ColorStop d_stopsGrayscale[2];
__device__ __constant__ ColorStop d_stopsBluePurleRed[3];

// const ColorStop h_stopsPythonLike[] = {
//         { 0.0f, Color3::init(0.2298057f, 0.29871797f, 0.75368315f) }, // Dark Blue
//         { 0.25f, Color3::init(0.23437708f, 0.30554173f, 0.75967953f) }, // Mid Blue
//         { 0.5f, Color3::init(0.27582712f, 0.36671692f, 0.81255294f) }, // White
//         { 0.75f, Color3::init(0.79606387f, 0.84869321f, 0.93347147f) }, // Light Orange
//         { 1.0f, Color3::init(0.70567316f, 0.01555616f, 0.15023281f) }  // Red
// };

// const ColorStop h_stopsPythonLike[] = {
//         { 0.0f, Color3::init(0.2298057f, 0.29871797f, 0.75368315f) }, // Dark Blue
//         { 0.85f, Color3::init(0.23437708f, 0.30554173f, 0.75967953f) }, // Mid Blue
//         { 0.90f, Color3::init(0.27582712f, 0.36671692f, 0.81255294f) }, // White
//         { 0.95f, Color3::init(0.79606387f, 0.84869321f, 0.93347147f) }, // Light Orange
//         { 1.0f, Color3::init(0.70567316f, 0.01555616f, 0.15023281f) }  // Red
// };

// Python "jet" colour scheme
const ColorStop h_stopsPythonLike[] = {
    { 0.00f, Color3::init(0.00000000f, 0.00000000f, 0.50000000f) },
    { 0.82f, Color3::init(0.00000000f, 0.00000000f, 0.94563280f) },
    { 0.84f, Color3::init(0.00000000f, 0.30000000f, 1.00000000f) },
    { 0.86f, Color3::init(0.00000000f, 0.69215686f, 1.00000000f) },
    { 0.88f, Color3::init(0.16129032f, 1.00000000f, 0.80645161f) },
    { 0.90f, Color3::init(0.49019608f, 1.00000000f, 0.47754586f) },
    { 0.92f, Color3::init(0.80645161f, 1.00000000f, 0.16129032f) },
    { 0.94f, Color3::init(1.00000000f, 0.77051561f, 0.00000000f) },
    { 0.96f, Color3::init(1.00000000f, 0.40740741f, 0.00000000f) },
    { 0.98f, Color3::init(0.94563280f, 0.02977487f, 0.00000000f) },
    { 1.00f, Color3::init(0.50000000f, 0.00000000f, 0.00000000f) },
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
__device__ Point3 d_lightPos;
__device__ Color3 d_backgroundColor;
__device__ Vec3 d_cameraUp;

Vec3 h_cameraUp = Vec3::init(0.0, 1.0, 0.0).normalize();


// Copy the above values to the device
void copyConstantsToDevice() {
    check_cuda_errors(cudaGetLastError());
    // ----------------------- Colour mapping -----------------------
    cudaMemcpyToSymbol(d_stopsPythonLike, h_stopsPythonLike, sizeof(h_stopsPythonLike));
    check_cuda_errors(cudaGetLastError());
    cudaMemcpyToSymbol(d_stopsGrayscale, h_stopsGrayscale, sizeof(h_stopsGrayscale));
    cudaMemcpyToSymbol(d_stopsBluePurleRed, h_stopsBluePurleRed, sizeof(h_stopsBluePurleRed));

    // ----------------------- Camera and Light -----------------------
    cudaMemcpyToSymbol(d_cameraUp, &h_cameraUp, sizeof(Vec3));
}


// ----------------------- TransferFunction -----------------------
__device__ float d_opacityK;
__device__ float d_sigmoidShift;
__device__ float d_sigmoidExp;
__device__ float d_alphaAcumLimit;
__device__ int d_tfComboSelected;
__device__ int d_tfComboSelectedColor;
__device__ float d_opacityConst;
__device__ bool d_showSilhouettes;
__device__ float d_silhouettesThreshold;

// ----------------------- Raycasting -----------------------
__device__ int d_samplesPerPixel;
