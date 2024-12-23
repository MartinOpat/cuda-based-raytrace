#include <netcdf>
#include <cassert>

using namespace std;
using namespace netCDF;

template <typename T>
void readAndAllocateAxis(T** axis_ptr, size_t *size, NcVar var) {
    assert(var.getDimCount() == 1);
    netCDF::NcDim dim = var.getDim(0);
    *size = dim.getSize();
    cudaError_t status = cudaMallocManaged(axis_ptr, *size*sizeof(T));
    var.getVar(*axis_ptr);
}