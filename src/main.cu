#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#include "hurricanedata/datareader.h"
#include "linalg/linalg.h" 
#include "objs/sphere.h"
#include "img/handler.h"
#include "consts.h"
#include "illumination/illumination.h"


static float* d_volume = nullptr;


void getTemperature(std::vector<float>& temperatureData, int idx = 0) {
    std::string path = "data/trimmed";
    std::string variable = "T";
    DataReader dataReader(path, variable);
    size_t dataLength = dataReader.fileLength(idx);
    temperatureData.resize(dataLength);
    dataReader.loadFile(temperatureData.data(), idx);
}

void getSpeed(std::vector<float>& speedData, int idx = 0) {
    std::string path = "data/trimmed";
    std::string varU = "U";
    std::string varV = "V";

    DataReader dataReaderU(path, varU);
    DataReader dataReaderV(path, varV);

    size_t dataLength = dataReaderU.fileLength(idx);
    speedData.resize(dataLength);
    std::vector<float> uData(dataLength);
    std::vector<float> vData(dataLength);

    dataReaderU.loadFile(uData.data(), idx);
    dataReaderV.loadFile(vData.data(), idx);

    for (int i = 0; i < dataLength; i++) {
        speedData[i] = sqrt(uData[i]*uData[i] + vData[i]*vData[i]);
    }
}

int main(int argc, char** argv) {
    std::vector<float> data;
    // getTemperature(data);
    getSpeed(data);


    // TODO: Eveontually remove debug below (i.e., eliminate for-loop etc.)
    // Generate debug volume data
    float* hostVolume = new float[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH];
    // generateVolume(hostVolume, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH);
    for (int i = 0; i < VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH; i++) {  // TODO: This is technically an unnecessary artifact of the old code taking in a float* instead of a std::vector
        // Discard temperatures above a small star (supposedly, missing temperature values)
        hostVolume[i] = data[i];
        if (data[i] + epsilon >= infty) hostVolume[i] = 0.0f;
    }

    // Min-max normalization
    float minVal = *std::min_element(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
    float maxVal = *std::max_element(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
    for (int i = 0; i < VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH; i++) {
        hostVolume[i] = (hostVolume[i] - minVal) / (maxVal - minVal);
    }

    // Allocate + copy data to GPU
    size_t volumeSize = sizeof(float) * VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH;
    cudaMalloc((void**)&d_volume, volumeSize);
    cudaMemcpy(d_volume, hostVolume, volumeSize, cudaMemcpyHostToDevice);

    // Allocate framebuffer
    unsigned char* d_framebuffer;
    size_t fbSize = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(unsigned char);
    cudaMalloc((void**)&d_framebuffer, fbSize);
    cudaMemset(d_framebuffer, 0, fbSize);

    // Copy external constants from consts.h to cuda
    copyConstantsToDevice();

    // Launch kernel
    dim3 blockSize(16, 16);  // TODO: Figure out a good size for parallelization
    dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1)/blockSize.x,
                  (IMAGE_HEIGHT + blockSize.y - 1)/blockSize.y);

    raycastKernel<<<gridSize, blockSize>>>(
        d_volume,
        d_framebuffer
    );
    cudaDeviceSynchronize();

    // Copy framebuffer back to CPU
    unsigned char* hostFramebuffer = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
    cudaMemcpy(hostFramebuffer, d_framebuffer, fbSize, cudaMemcpyDeviceToHost);

    // Export image
    saveImage("output.ppm", hostFramebuffer, IMAGE_WIDTH, IMAGE_HEIGHT);

    // Cleanup
    delete[] hostVolume;
    delete[] hostFramebuffer;
    cudaFree(d_volume);
    cudaFree(d_framebuffer);

    std::cout << "Phong-DVR rendering done. Image saved to output.ppm" << std::endl;
    return 0;
}