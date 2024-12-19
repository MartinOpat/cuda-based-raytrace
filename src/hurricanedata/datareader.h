#ifndef DATAREADER_H
#define DATAREADER_H

#include <vector>
#include <string>

std::vector<float> readData(std::string path, std::string variableName);
struct cudaArray* loadDataToDevice(std::string path, std::string variableName);

#endif //DATAREADER_H
