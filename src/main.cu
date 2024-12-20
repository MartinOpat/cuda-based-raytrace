#include "hurricanedata/datareader.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip> 
#include <cmath>


int main() {
    std::string path = "data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4";
    std::string variable = "T";
    auto x = readData(path, variable);

    // Calculate the mean of the data to see if it works.
    float sum = 0;
    int n = 0;
    int skipped = 0;
    for(int i = 0; i < x.size(); i++) {
        if (x[i] < 1E14) {
            sum += x[i];
            n++;
        } else {
            skipped++;
        }
    }
    std::cout << "Mean = " << sum/n << " and sum = " << sum << " using " << n << " values in computation and skipped " << skipped << " values.\n";

    return 0;
}
