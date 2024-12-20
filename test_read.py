import numpy as np
from netCDF4 import Dataset
from math import prod

# Load the NetCDF file
file_path = 'data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4'
ncfile = Dataset(file_path, 'r')

# Check the available variables in the file
print(ncfile.variables.keys())

U = ncfile.variables['T'][:] 

# Check the shape of the variable
print(f"Shape of U: {U.shape} and total length is {prod(U.shape)}")

# Compute the mean of the variable across all axes (for all elements in U)
U_mean = np.mean(U)

# Print the mean
print("Mean of U:", U_mean)

print(f"{U[0,0,0,1]=}")
is_masked = np.ma.isMaskedArray(U)
print(f"Is U a masked array? {is_masked}")
masked_count = np.ma.count_masked(U)
print("Number of masked values in U:", masked_count)

# Close the NetCDF file
ncfile.close()
