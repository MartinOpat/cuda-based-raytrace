#include <algorithm>
#include <cmath>
#include "consts.h"
#include <cuda_runtime.h>
#include <fstream>
#include "gui/MainWindow.h"
#include "hurricanedata/datareader.h"
#include "illumination/illumination.h"
#include "img/handler.h"
#include <iostream>
#include "linalg/linalg.h" 
#include <vector>
#include <numeric>


static float* d_volume = nullptr;

// TODO: general
// * pass camera_info to the raycasting function - updated according to glfw.
// * on that note, code for handling input (mouse movement certainly, possibly free input / 4 pre-coded views, q/esc to quit, space for pause (would be were the 'simple' render idea would come in))
// * very similarly - actual code for loading new data as the simulation progresses - right now its effectively a static image loader

void getTemperature(std::vector<float>& temperatureData, int idx = 0) {
    std::string path = "data/trimmed";
    // std::string path = "data";
    std::string variable = "T";
    DataReader dataReader(path, variable);
    size_t dataLength = dataReader.fileLength(idx);
    temperatureData.resize(dataLength);
    dataReader.loadFile(temperatureData.data(), idx);
}

void getSpeed(std::vector<float>& speedData, int idx = 0) {
    std::string path = "data/trimmed";
    // std::string path = "data";
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


int main() {
  std::vector<float> data;
  getTemperature(data, 0);
  // getSpeed(data, 294);

  std::cout << "DATA size: " << data.size() << std::endl;

  // TODO: Eveontually remove debug below (i.e., eliminate for-loop etc.)
  // Generate debug volume data
  float* hostVolume = new float[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH];
  // generateVolume(hostVolume, VOLUME_WIDTH, VOLUME_HEIGHT, VOLUME_DEPTH);
  int inftyCount=0;
  for (int i = 0; i < VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH; i++) {
    // Discard temperatures above a small star (supposedly, missing temperature values)
    hostVolume[i] = data[i + 0*VOLUME_DEPTH*VOLUME_HEIGHT*VOLUME_WIDTH];
    if (data[i + 0*VOLUME_DEPTH*VOLUME_HEIGHT*VOLUME_WIDTH] + epsilon >= infty) {hostVolume[i] = -infty; inftyCount++;}
  }
  std::cout << "inftyCount: " << inftyCount << std::endl;

  // Reverse the order of hostVolume
  for (int i = 0; i < VOLUME_WIDTH; i++) {
    for (int j = 0; j < VOLUME_HEIGHT; j++) {
      for (int k = 0; k < VOLUME_DEPTH/2; k++) {
        float temp = hostVolume[i + j*VOLUME_WIDTH + k*VOLUME_WIDTH*VOLUME_HEIGHT];
        hostVolume[i + j*VOLUME_WIDTH + k*VOLUME_WIDTH*VOLUME_HEIGHT] = hostVolume[i + j*VOLUME_WIDTH + (VOLUME_DEPTH - 1 - k)*VOLUME_WIDTH*VOLUME_HEIGHT];
        hostVolume[i + j*VOLUME_WIDTH + (VOLUME_DEPTH - 1 - k)*VOLUME_WIDTH*VOLUME_HEIGHT] = temp;
      }
    }
  }


  // Store the half-way up slice data into a file
  std::ofstream myfile;
  myfile.open("halfwayup.txt");
  for (int i = 0; i < VOLUME_WIDTH; i++) {
    for (int j = 0; j < VOLUME_HEIGHT; j++) {
      myfile << hostVolume[i + j*VOLUME_WIDTH + VOLUME_DEPTH/2*VOLUME_WIDTH*VOLUME_HEIGHT] << " ";
    }
    myfile << std::endl;
  }
  myfile.close();

  float minVal = *std::min_element(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH, [](float a, float b) {
    if (a <= epsilon) return false;
    if (b <= epsilon) return true;
    return a < b;
  });
  float maxVal = *std::max_element(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
  std::cout << "minVal: " << minVal << " maxVal: " << maxVal << std::endl;

  // Min-max normalization TODO: Decide whether to keep the normalization here but probably not
  // Normalize to [0, 1]
  // Temperature: min: 0 max: 1 avg: 0.776319 median: 0.790567
  // Speed: min: 0 max: 1 avg: 0.132117 median: 0.0837869
  // for (int i = 0; i < VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH; i++) {
  //   hostVolume[i] = (hostVolume[i] - minVal) / (maxVal - minVal);
  // }

  // // print min, max, avg., and median values <--- the code actually does not work when this snippet is enabled so probably TODO: Delete this later
  // std::sort(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
  // float sum = std::accumulate(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH, 0.0f);
  // float avg = sum / (VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
  // std::cout << "min: " << hostVolume[0] << " max: " << hostVolume[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH - 1] << " avg: " << avg << " median: " << hostVolume[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH / 2] << std::endl;

  // Allocate + copy data to GPU
  size_t volumeSize = sizeof(float) * VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH;
  cudaMalloc((void**)&d_volume, volumeSize);
  cudaMemcpy(d_volume, hostVolume, volumeSize, cudaMemcpyHostToDevice);

  // Allocate framebuffer
  // unsigned char* d_framebuffer;
  // size_t fbSize = IMAGE_WIDTH * IMAGE_HEIGHT * 3 * sizeof(unsigned char);
  // cudaMalloc((void**)&d_framebuffer, fbSize);
  // cudaMemset(d_framebuffer, 0, fbSize);

  // Copy external constants from consts.h to cuda
  copyConstantsToDevice();

  // NOTE: this is done within the rayTracer class
  // // Launch kernel
  // dim3 blockSize(16, 16);  
  // dim3 gridSize((IMAGE_WIDTH + blockSize.x - 1)/blockSize.x,
  //               (IMAGE_HEIGHT + blockSize.y - 1)/blockSize.y);
  //
  // raycastKernel<<<gridSize, blockSize>>>(
  //     d_volume,
  //     d_framebuffer
  // );
  // cudaDeviceSynchronize();

  Window window(IMAGE_WIDTH, IMAGE_HEIGHT);
  int out = window.init(d_volume);

  cudaFree(d_volume);

  // // Copy framebuffer back to CPU
  // unsigned char* hostFramebuffer = new unsigned char[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
  // cudaMemcpy(hostFramebuffer, d_framebuffer, fbSize, cudaMemcpyDeviceToHost);
  //
  // // Export image
  // saveImage("output.ppm", hostFramebuffer, IMAGE_WIDTH, IMAGE_HEIGHT);
  //
  // // Cleanup //TODO: cleanup properly
  delete[] hostVolume;
  // delete[] hostFramebuffer;
  // cudaFree(d_volume);
  // cudaFree(d_framebuffer);
  //
  // std::cout << "Phong-DVR rendering done. Image saved to output.ppm" << std::endl;
  // return 0;
  return out;
}
