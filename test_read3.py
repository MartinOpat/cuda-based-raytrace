import numpy as np
from netCDF4 import Dataset

# file_path = 'data/MERRA2_400.inst6_3d_ana_Np.20120101.nc4'
# ncfile = Dataset(file_path, 'r')

file_paths = [
    'data/atmosphere_MERRA-wind-speed[179253532]/MERRA2_400.inst6_3d_ana_Np.20120101.nc4',
    'data/atmosphere_MERRA-wind-speed[179253532]/MERRA2_400.inst6_3d_ana_Np.20120102.nc4',
    'data/atmosphere_MERRA-wind-speed[179253532]/MERRA2_400.inst6_3d_ana_Np.20120103.nc4'
]

ncfiles = [Dataset(file_path) for file_path in file_paths]


# print(f"{Temp[0, 20, 100, 100]=}")

for i in range(10):
    Temp = ncfiles[i//4].variables['T'][:] 
    x = Temp[i%4, 20, 100, 100]

    Temp2 = ncfiles[(i+1)//4].variables['T'][:] 
    y = Temp2[(i+1)%4, 20, 100, 100]
    print(f"{(x+y)/2=}")



# Close the NetCDF file
for ncfile in ncfiles:
    ncfile.close()