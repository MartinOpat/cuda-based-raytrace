#pragma once

#include <cuda_runtime.h>
#include <cmath>

struct Vec3 {  // TODO: Maybe make this into a class ... maybe
    double x, y, z;

    static __host__ __device__ Vec3 init(double x, double y, double z) {Vec3 v = {x, y, z}; return v;}
    static __host__ __device__ Vec3 zero() { return Vec3::init(0, 0, 0); }
    static __host__ __device__ Vec3 init() { return zero(); }
    static __host__ __device__ Vec3 init(const double (&arr)[3]) { return Vec3::init(arr[0], arr[1], arr[2]); }

    __host__ __device__ Vec3 operator+(const Vec3& b) const { return Vec3::init(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ Vec3& operator+=(const Vec3& b) { x += b.x; y += b.y; z += b.z; return *this; }
    
    __host__ __device__ Vec3 operator-() const { return Vec3::init(-x, -y, -z); }
    __host__ __device__ Vec3 operator-(const Vec3& b) const { return Vec3::init(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ Vec3& operator-=(const Vec3& b) { x -= b.x; y -= b.y; z -= b.z; return *this; }
    
    __host__ __device__ Vec3 operator*(double b) const { return Vec3::init(x * b, y * b, z * b); }
    __host__ __device__ Vec3& operator*=(double b) { x *= b; y *= b; z *= b; return *this; }

    __host__ __device__ double dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 cross(const Vec3& b) const { return Vec3::init(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
    __host__ __device__ Vec3 normalize() const { double len = sqrt(x * x + y * y + z * z); return Vec3::init(x / len, y / len, z / len); }
};

typedef Vec3 Point3;
typedef Vec3 Color3;
