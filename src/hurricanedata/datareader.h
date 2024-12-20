#ifndef DATAREADER_H
#define DATAREADER_H

#include <vector>
#include <string>

std::vector<float> readData(std::string path, std::string variableName);
std::pair<float*, size_t> loadDataToDevice(std::string path, std::string variableName);

#endif //DATAREADER_H
