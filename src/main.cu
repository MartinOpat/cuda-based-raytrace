#include "consts.h"
#include <cuda_runtime.h>
#include "gui/MainWindow.h"
#include "hurricanedata/datareader.h"
#include <iostream>
#include <vector>


static float* d_volume = nullptr;

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
  getTemperature(data, 254); // 20121028
  // getSpeed(data, 254); // 20121028

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
