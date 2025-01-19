#pragma once

#include <cuda_runtime.h>
#include <cmath>

struct Vec3 {
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
    __host__ __device__ Vec3 operator*(float b) const { return Vec3::init(x * b, y * b, z * b); }
    __host__ __device__ Vec3& operator*=(float b) { x *= b; y *= b; z *= b; return *this; }
    friend __host__ __device__ Vec3 operator*(float a, const Vec3& b) {
        return Vec3::init(a * b.x, a * b.y, a * b.z);
    }


    __host__ __device__ double dot(const Vec3& b) const { return x * b.x + y * b.y + z * b.z; }
    __host__ __device__ Vec3 cross(const Vec3& b) const { return Vec3::init(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
    __host__ __device__ Vec3 normalize() const { double len = sqrt(x * x + y * y + z * z); return Vec3::init(x / len, y / len, z / len); }
    __host__ __device__ double length() const { return sqrt(x * x + y * y + z * z); }

    __host__ __device__ void setDirectionFromEuler(double pitch, double yaw, double roll) {
        // Compute the direction vector using the Euler angles in radians
        double cosPitch = cos(pitch);
        double sinPitch = sin(pitch);
        double cosYaw = cos(yaw);
        double sinYaw = sin(yaw);

        // Direction vector components
        x = cosPitch * cosYaw;
        y = cosPitch * sinYaw;
        z = sinPitch;
    }

    static __host__ __device__ Vec3 getDirectionFromEuler(double pitch, double yaw, double roll) {
        Vec3 v = Vec3::init(1,0,0);
        v.setDirectionFromEuler(pitch, yaw, roll);
        return v;
    }

    __host__ __device__ void rotateAroundAxis(const Vec3& axis, double angle) {
        double cosA = cos(angle);
        double sinA = sin(angle);

        Vec3 rotated = *this * cosA + axis.cross(*this) * sinA + axis * axis.dot(*this) * (1 - cosA);
        *this = rotated;
    }
};

typedef Vec3 Point3;
typedef Vec3 Color3;
