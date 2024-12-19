#include "hurricanedata/datareader.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

int main() {
    std::string path = "data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4";
    std::string variable = "U";
    auto arr = loadDataToDevice(path, variable);
    cudaFreeArray(arr);

    return 0;
}
