#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <cmath>

#define WIDTH 3840
#define HEIGHT 2160
#define SAMPLES_PER_PIXEL 8

struct Vec3 {
    double x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(double x, double y, double z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3 operator*(double b) const { return Vec3(x * b, y * b, z * b); }
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ double dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 normalize() const { double len = sqrt(x * x + y * y + z * z); return Vec3(x / len, y / len, z / len); }
};

struct Sphere {
    Vec3 center;
    double radius;
    Vec3 color;

    __device__ bool intersect(const Vec3& rayOrigin, const Vec3& rayDir, double& t) const {
        Vec3 oc = rayOrigin - center;
        double b = oc.dot(rayDir);
        double c = oc.dot(oc) - radius * radius;
        double h = b * b - c;
        if (h < 0.0) return false;
        h = sqrt(h);
        t = -b - h;
        return true;
    }
};

__device__ Vec3 phongShading(const Vec3& point, const Vec3& normal, const Vec3& lightDir, const Vec3& viewDir, const Vec3& color) {
    double ambientStrength = 0.1;
    double diffuseStrength = 0.8;
    double specularStrength = 0.5;
    int shininess = 64;

    Vec3 ambient = color * ambientStrength;
    double diff = max(normal.dot(lightDir), 0.0);
    Vec3 diffuse = color * (diffuseStrength * diff);

    Vec3 reflectDir = (normal * (2.0 * normal.dot(lightDir)) - lightDir).normalize();
    double spec = pow(max(viewDir.dot(reflectDir), 0.0), shininess);
    Vec3 specular = Vec3(1.0, 1.0, 1.0) * (specularStrength * spec);

    return ambient + diffuse + specular;
}

__global__ void renderKernel(unsigned char* framebuffer, Sphere* spheres, int numSpheres, Vec3 lightPos) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    int pixelIndex = (y * WIDTH + x) * 3;
    Vec3 rayOrigin(0, 0, 0);
    Vec3 colCum(0, 0, 0);

    double spp = static_cast<double>(SAMPLES_PER_PIXEL);
    for (int sample = 0; sample < SAMPLES_PER_PIXEL; sample++) {
        double u = (x + (sample / spp) - WIDTH / 2.0) / WIDTH;
        double v = (y + (sample / spp) - HEIGHT / 2.0) / HEIGHT;
        Vec3 rayDir(u, v, 1.0);
        rayDir = rayDir.normalize();

        for (int i = 0; i < numSpheres; ++i) {
            double t;
            if (spheres[i].intersect(rayOrigin, rayDir, t)) {
                Vec3 hitPoint = rayOrigin + rayDir * t;
                Vec3 normal = (hitPoint - spheres[i].center).normalize();
                Vec3 lightDir = (lightPos - hitPoint).normalize();
                Vec3 viewDir = -rayDir;

                colCum = colCum + phongShading(hitPoint, normal, lightDir, viewDir, spheres[i].color);
            }
        }
    }

    // Average color across all samples
    Vec3 color = colCum * (1.0 / SAMPLES_PER_PIXEL);

    framebuffer[pixelIndex] = static_cast<unsigned char>(fmin(color.x, 1.0) * 255);
    framebuffer[pixelIndex + 1] = static_cast<unsigned char>(fmin(color.y, 1.0) * 255);
    framebuffer[pixelIndex + 2] = static_cast<unsigned char>(fmin(color.z, 1.0) * 255);
}

void saveImage(const char* filename, unsigned char* framebuffer) {
    std::ofstream imageFile(filename, std::ios::out | std::ios::binary);
    imageFile << "P6\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int i = 0; i < WIDTH * HEIGHT * 3; i++) {
        imageFile << framebuffer[i];
    }
    imageFile.close();
}

int main() {
    Sphere spheres[] = {
        { Vec3(0, 0, 5), 1.0, Vec3(1.0, 0.0, 0.0) },  // Red sphere
        { Vec3(-2, 1, 7), 1.0, Vec3(0.0, 1.0, 0.0) }, // Green sphere
        { Vec3(2, -1, 6), 1.0, Vec3(0.0, 0.0, 1.0) }  // Blue sphere
    };
    int numSpheres = sizeof(spheres) / sizeof(Sphere);
    Vec3 lightPos(5, 5, 0);

    unsigned char* d_framebuffer;
    unsigned char* h_framebuffer = new unsigned char[WIDTH * HEIGHT * 3];
    Sphere* d_spheres;
    cudaMalloc(&d_framebuffer, WIDTH * HEIGHT * 3);
    cudaMalloc(&d_spheres, numSpheres * sizeof(Sphere));
    cudaMemcpy(d_spheres, spheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    renderKernel<<<numBlocks, threadsPerBlock>>>(d_framebuffer, d_spheres, numSpheres, lightPos);
    cudaDeviceSynchronize();

    cudaMemcpy(h_framebuffer, d_framebuffer, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);
    saveImage("output.ppm", h_framebuffer);

    cudaFree(d_framebuffer);
    cudaFree(d_spheres);
    delete[] h_framebuffer;

    std::cout << "High-resolution image saved as output.ppm" << std::endl;
    return 0;
}
