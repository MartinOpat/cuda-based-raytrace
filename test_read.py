import numpy as np
from netCDF4 import Dataset

# Load the NetCDF file
file_path = 'data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4'
ncfile = Dataset(file_path, 'r')

# Check the available variables in the file
print(ncfile.variables.keys())

U = ncfile.variables['T'][:] 

# Check the shape of the variable 
print("Shape of U:", U.shape)

# Compute the mean of the variable across all axes (for all elements in U)
U_mean = np.mean(U)
U_sum = np.sum(U)

# Print the mean
print("Mean of U:", U_mean)

print("Sum of U:", U_sum)

sumval = 0
row = U[2,20,100]
n = 0
for val in row:
    if not np.ma.is_masked(val):
        n+=1
        sumval += val
print(f"Why does {np.mean(row)=} not equal {sumval/n=} ?!")

# Close the NetCDF file
ncfile.close()