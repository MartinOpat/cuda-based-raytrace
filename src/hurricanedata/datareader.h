#ifndef DATAREADER_H
#define DATAREADER_H

#include <string>
#include <experimental/propagate_const>
#include <memory>

#include "filepathmanager.h"


class DataReader {
public:
    DataReader(const std::string &path, std::string variableName);
    size_t fileLength(size_t fileIndex);

    template <typename T>
    void loadFile(T* dataOut, size_t fileIndex);

    ~DataReader();
private:
    FilePathManager filePathManager;
    std::string variableName;
};


#endif //DATAREADER_H
