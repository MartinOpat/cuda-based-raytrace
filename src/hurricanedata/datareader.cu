#include "datareader.h"

#include <netcdf>

using namespace std;
using namespace netCDF;

std::vector<float> readData(std::string path, std::string variableName) {
    netCDF::NcFile data(path, netCDF::NcFile::read);

    multimap<string, NcVar> vars = data.getVars();

    NcVar var = vars.find(variableName)->second;   

    int length = 1;
    for (NcDim dim: var.getDims()) {
        length *= dim.getSize();
    }

    vector<float> vec(length);

    var.getVar(vec.data());

    return vec;
}