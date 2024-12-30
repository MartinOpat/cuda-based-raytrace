#pragma once

#include <cuda_runtime.h>
#include <cmath>

struct Vec3 {  // TODO: Maybe make this into a class
    double x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    __host__ __device__ Vec3(const double (&arr)[3]) : x(arr[0]), y(arr[1]), z(arr[2]) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    
    __host__ __device__ Vec3 operator-() const { return Vec3(-x, -y, -z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3& operator-=(const Vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    
    __host__ __device__ Vec3 operator*(double b) const { return Vec3(x * b, y * b, z * b); }
    __host__ __device__ Vec3& operator*=(double b) { x *= b; y *= b; z *= b; return *this; }

    __host__ __device__ double dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 cross(const Vec3& b) const { return Vec3(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
    __host__ __device__ Vec3 normalize() const { double len = sqrt(x * x + y * y + z * z); return Vec3(x / len, y / len, z / len); }
};