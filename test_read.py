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

masked_count = np.ma.count_masked(U)
print("Number of masked values in U:", masked_count)

nan_count = np.isnan(U).sum()
print("Number of NaN values in U:", nan_count)

print("Calculating mean manually (takes a bit cause python is slowww)")

count = 0
valsum = 0
for val in U.flat:
    if not np.ma.is_masked(val):
        # print(val)
        valsum += val
        count += 1

print(f"{valsum=} {valsum/count=} {count=}")

print(f"The problem is this: why does {valsum/count=} not equal {U_mean=}")

# Close the NetCDF file
ncfile.close()
