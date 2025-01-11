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


static float* d_volume = nullptr;

// FIXME: segfaults on window resize - the raycasting function should work with window->w and window-h instead of constants.

// TODO: general
// * very similarly - actual code for loading new data as the simulation progresses - right now its effectively a static image loader * pause button once that dataloading is implemented 

// * save frames to file while running program -> then export to gif on close.
// * time controls - arbitrary skipping to specified point (would require some changes to gpubuffer) (could have)

// * transfer function -> move the code in raycastkernel to its own class and add silhouette detection here as well.

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


int main() {
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

  // Create the GUI
  Window window(IMAGE_WIDTH, IMAGE_HEIGHT);
  int out = window.init(d_volume);

  // memory management
  cudaFree(d_volume);
  delete[] hostVolume;
  return out;
}
