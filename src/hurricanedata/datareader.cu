#include "datareader.h"

#include <netcdf>
#include <optional>

using namespace std;
using namespace netCDF;

DataReader::DataReader(const std::string &path, std::string variableName):
filePathManager(path), variableName(variableName) { }

size_t DataReader::fileLength(size_t fileIndex) {
    cout << "filePathMan = " << filePathManager.getPath(fileIndex) << " bla = " << fileIndex << "\n";

    NcFile data(filePathManager.getPath(fileIndex), NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(variableName)->second;   
    
    size_t length = 1;
    for (NcDim dim: var.getDims()) {
        length *= dim.getSize();
    }

    // TODO: Turns out c-NetCDF is not thread safe :( https://github.com/Unidata/netcdf-c/issues/1373
    // size_t length = 34933248;
    std::cout << "return length" << length << "\n";
    return length;
}

template <typename T>
void DataReader::loadFile(T* dataOut, size_t fileIndex) {
    std::cout << "loading file" << fileIndex <<"\n";
    NcFile data(filePathManager.getPath(fileIndex), NcFile::read);

    // multimap<string, NcVar> vars = data.getVars();

    // NcVar var = vars.find(variableName)->second;   

    // var.getVar(dataOut);
}

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