<br />
<p align="center">
  <h1 align="center">CUDA-based Raycaster</h1>

  <p align="center">
    This project implements a raycaster on nvidia GPUs, visualizing wind patterns through the use of silhouette shading
  </p> ## Table of Contents

* [About the Project](#about) 
* [Dependencies](#dependencies)
* [Installation](#installation)
  * [Running](#running)
  * [Screenshots](#screenshots) 
* [Modules](#modules) 


## About

This project was developed as part of the course Geo-Visualization at the University of Groningen.
It contains a CUDA-based raycaster which visualizes wind patterns using a silhouette shading.
The program also includes various configuration options, which can be set as the program runs (see [Screenshots](#screenshots)).
The program expects to use the MERRA2 dataset, which can be found [here](https://disc.gsfc.nasa.gov/datasets/M2I6NPANA_5.12.4/summary?keywords=M2I6NPANA_5.12.4).


## Dependencies

The project depends on a few libraries.
Below is a list of these along with versions that _will_ work.
Newer/older versions of the libraries may also compile and run properly, but this has not been tested.
 * OpenGL[^1] Tested using the Mesa (1:24.3.3-2) driver.
 * glfw3 (3.4-2)
 * Cuda (12.6.3-1)
 * Netcdf (4.9.2-6)
 * Netcdf-cxx (4.3.1-4)
 * Dear ImGUI (submodule within repository)


## Installation

Once all libraries are installed compiling the program is quite straightforward. First, initialize the Dear ImGUI submodule:
```bash
git submodule update --init
```

Then compile using CMake:
```bash
mkdir build
cd build
cmake ..
make 
./cuda-raytracer
```
Make sure the program is executed on the nvidia GPU, or it will crash on startup.
For example, on Arch the command `prime-run ./cuda-raytracer` would make sure the program utilizes the GPU.

## Running

### A note on data loading

In order to run properly, the program expects to find a number of `.nc4` files in a directory relative to the executable. Specifically, the data should consist of a number of `.nc4` files from the MERRA2 dataset, located in the `./data/trimmed/` folder.
For best results - that is, to ensure no crashes during runtime - this folder should contain a full year of data. 
The program loads a specific data file based on index, meaning the files should be named in alphabetical order.

For example, using the date of the file in question is a good way of ensuring this:
 * MERRA2_400.inst6_3d_ana_Np.20120101.nc4
 * MERRA2_400.inst6_3d_ana_Np.20120102.nc4
 * MERRA2_400.inst6_3d_ana_Np.20120103.nc4
 * MERRA2_400.inst6_3d_ana_Np.20120104.nc4


### Actual execution

Once the data is in place, the program may be executed as normal - again make sure to run this on an NVidia GPU.  

## Screenshots

Temperature of hurricane Sandy visualized using the raycaster over time:

![alt-text](figures/hurricane_flipped.gif)


