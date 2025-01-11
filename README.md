# cuda-based-raytrace

## How to run

First, initialize the imGui submodule:
```bash
git submodule init imgui
git submodule update imgui
```

Then, compile using cmake:
```bash
mkdir build
cd build
cmake ..
make 
./cuda-raytracer
```
