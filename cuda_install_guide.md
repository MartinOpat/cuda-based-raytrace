# Assuminng NVIDIA drivers present

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run
chmod +x cuda_12.5.1_555.42.06_linux.run
sudo ./cuda_12.5.1_555.42.06_linux.run
```

Note: Make sure that the cuda version (above 555) matches the driver version. Newest cuda drivers can be downloaded from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). 

Additionally, disabling the GUI via `ctrl+alt+f3` and subsequently running:

```bash
sudo systemctl isolate multi-user.target
```

is recommended.