#include "datareader.h"

#include <netcdf>
#include <cassert>

using namespace std;
using namespace netCDF;

DataReader::DataReader(const std::string &path, std::string variableName):
filePathManager(path), variableName(variableName) { }

size_t DataReader::fileLength(size_t fileIndex) {
    NcFile data(filePathManager.getPath(fileIndex), NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(variableName)->second;   
    
    size_t length = 1;
    for (NcDim dim: var.getDims()) {
        length *= dim.getSize();
    }

    return length;
}

size_t DataReader::axisLength(size_t fileIndex, const std::string& axisName) {
    NcFile data(filePathManager.getPath(fileIndex), NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(axisName)->second;   

    assert(var.getDimCount() == 1);

    netCDF::NcDim dim = var.getDim(0);
    return dim.getSize();
}

template <typename T>
void DataReader::loadFile(T* dataOut, size_t fileIndex) {
    loadFile(dataOut, fileIndex, variableName);
}

template <typename T>
void DataReader::loadFile(T* dataOut, size_t fileIndex, const string& varName) {
    NcFile data(filePathManager.getPath(fileIndex), NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(varName)->second;   

    var.getVar(dataOut);
}

template void DataReader::loadFile<float>(float* dataOut, size_t fileIndex, const string& variableName);
template void DataReader::loadFile<int>(int* dataOut, size_t fileIndex, const string& variableName);
template void DataReader::loadFile<double>(double* dataOut, size_t fileIndex, const string& variableName);
template void DataReader::loadFile<double>(double* dataOut, size_t fileIndex);
template void DataReader::loadFile<float>(float* dataOut, size_t fileIndex);
template void DataReader::loadFile<int>(int* dataOut, size_t fileIndex);

DataReader::~DataReader() {

}

// template <typename T>
// void DataReader::readAndAllocateAxis(T** axis_ptr, size_t *size, const string &varName) {
//     assert(var.getDimCount() == 1);
//     netCDF::NcDim dim = var.getDim(0);
//     *size = dim.getSize();
//     cudaError_t status = cudaMallocManaged(axis_ptr, *size*sizeof(T));
//     var.getVar(*axis_ptr);
// }