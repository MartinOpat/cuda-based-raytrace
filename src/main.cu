#include "hurricanedata/datareader.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

int main() {
    std::string path = "data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4";
    std::string variable = "U";
    auto x = readData(path, variable);

    // Print some values from the file to see that it worked
    int num = 0;
    for(int i = 0; i < x.size(); i++) {
        if (x[i] < 1E14) std::cout << x[i] << "\n";
        if(num > 10000) break;
        num++;
    }

    return 0;
}
