import numpy as np
from netCDF4 import Dataset

# Load the NetCDF file
file_path = 'data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4'
ncfile = Dataset(file_path, 'r')

# Check the available variables in the file
print(ncfile.variables.keys())

Temp = ncfile.variables['T'][:] 

print(f"{Temp[1, 20, 100, 100]=}")
print(f"{Temp.flat[12949732]=}")


# Close the NetCDF file
ncfile.close()