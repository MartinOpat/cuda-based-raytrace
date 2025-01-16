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
// * actual code for loading new data as the simulation progresses - right now its effectively a static image loader
// * save frames to file while running program -> then export to gif on close.
// * time controls - arbitrary skipping to specified point (would require some changes to gpubuffer) (could have)

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

  // TODO: Eventually, we should not need to load the volume like this
  float* hostVolume = new float[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH];
  for (int i = 0; i < VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH; i++) {
    hostVolume[i] = data[i + 0*VOLUME_DEPTH*VOLUME_HEIGHT*VOLUME_WIDTH];
    // Discard missing values
    if (data[i + 0*VOLUME_DEPTH*VOLUME_HEIGHT*VOLUME_WIDTH] + epsilon >= infty) hostVolume[i] = -infty;
  }

  // Reverse the order of hostVolume - why is it upside down anyway?
  for (int i = 0; i < VOLUME_WIDTH; i++) {
    for (int j = 0; j < VOLUME_HEIGHT; j++) {
      for (int k = 0; k < VOLUME_DEPTH/2; k++) {
        float temp = hostVolume[i + j*VOLUME_WIDTH + k*VOLUME_WIDTH*VOLUME_HEIGHT];
        hostVolume[i + j*VOLUME_WIDTH + k*VOLUME_WIDTH*VOLUME_HEIGHT] = hostVolume[i + j*VOLUME_WIDTH + (VOLUME_DEPTH - 1 - k)*VOLUME_WIDTH*VOLUME_HEIGHT];
        hostVolume[i + j*VOLUME_WIDTH + (VOLUME_DEPTH - 1 - k)*VOLUME_WIDTH*VOLUME_HEIGHT] = temp;
      }
    }
  }

  // // Store the half-way up slice data into a file TODO: Remove this debug
  // std::ofstream myfile;
  // myfile.open("halfwayup.txt");
  // for (int i = 0; i < VOLUME_WIDTH; i++) {
  //   for (int j = 0; j < VOLUME_HEIGHT; j++) {
  //     myfile << hostVolume[i + j*VOLUME_WIDTH + VOLUME_DEPTH/2*VOLUME_WIDTH*VOLUME_HEIGHT] << " ";
  //   }
  //   myfile << std::endl;
  // }
  // myfile.close();

  // // Print min, max, avg., and median values TODO: Remove this debug
  // float minVal = *std::min_element(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH, [](float a, float b) {
  //   if (a <= epsilon) return false;
  //   if (b <= epsilon) return true;
  //   return a < b;
  // });
  // float maxVal = *std::max_element(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
  // std::cout << "minVal: " << minVal << " maxVal: " << maxVal << std::endl;
  // // print min, max, avg., and median values <--- the code actually does not work when this snippet is enabled so probably TODO: Delete this later
  // std::sort(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
  // float sum = std::accumulate(hostVolume, hostVolume + VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH, 0.0f);
  // float avg = sum / (VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH);
  // std::cout << "min: " << hostVolume[0] << " max: " << hostVolume[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH - 1] << " avg: " << avg << " median: " << hostVolume[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH / 2] << std::endl;

  // Allocate + copy data to GPU
  size_t volumeSize = sizeof(float) * VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH;
  cudaMalloc((void**)&d_volume, volumeSize);
  cudaMemcpy(d_volume, hostVolume, volumeSize, cudaMemcpyHostToDevice);

  copyConstantsToDevice();
  // Create the GUI
  Window window(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT);
  int out = window.init(d_volume);

  // memory management
  cudaFree(d_volume);
  delete[] hostVolume;
  return out;
}
