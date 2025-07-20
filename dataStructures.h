#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <cstdlib>

using uchar = unsigned char;
using uint = unsigned int;

struct vec3
{
    float x;
    float y;
    float z;

    __host__ __device__ vec3 operator + (const vec3& other) const // vec3 + vec3 (addition)
    {
        return { x + other.x, y + other.y, z + other.z };
    }

    __host__ __device__ vec3 operator - (const vec3& other) const // vec3 - vec3 (subtraction)
    {
        return { x - other.x, y - other.y, z - other.z };
    }

    __host__ __device__ vec3 operator * (const vec3& other) const // vec3 * vec3 (cross product)
    {
        return { y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x };
    }

    __host__ __device__ vec3 operator * (const float& scalar) const // vec3 * float (scalar multiplication)
    {
        return { x * scalar, y * scalar, z * scalar };
    }
};

__host__ __device__ inline vec3 normalize(const vec3& a)
{
    float dist = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    return vec3{ a.x / dist, a.y / dist, a.z / dist };
}

__host__ __device__ inline float dot(const vec3& a, const vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float randf(float min, float max)
{
    return (((float)rand() / (float)RAND_MAX) * (max - min)) + min;
}


struct color3
{
    float r;
    float g;
    float b;

    __host__ __device__ color3 operator * (const color3& other) const // multiply colors
    {
        return
        {
            r * other.r,
            g * other.g,
            b * other.b
        };
    }

    __host__ __device__ color3 operator + (const color3& other) const // add colors
    {
        return
        {
            r + other.r,
            g + other.g,
            b + other.b
        };
    }

    __host__ __device__ color3 operator / (const float& other) const // divide color by float
    {
        return
        {
            r / other,
            g / other,
            b / other
        };
    }

    __host__ __device__ color3 operator * (const float& other) const // multiply color by float
    {
        return
        {
            r * other,
            g * other,
            b * other
        };
    }

    __host__ __device__ color3& operator += (const color3& other)
    {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    __host__ __device__ color3& operator *= (const color3& other)
    {
        r *= other.r;
        g *= other.g;
        b *= other.b;
        return *this;
    }

    __host__ __device__ color3& operator *= (const float& other)
    {
        r *= other;
        g *= other;
        b *= other;
        return *this;
    }

    __host__ __device__ color3& operator /= (const float& other)
    {
        r /= other;
        g /= other;
        b /= other;
        return *this;
    }
};

struct camera
{
    vec3 position;
    vec3 direction;
    vec3 up;
    vec3 right;
    float depth;
};

struct sphere
{
    vec3 position;
    float radius;
    color3 color;
    bool isLightSource;
};

struct hitInfo
{
    bool didHit;
    bool didHitLight;
    vec3 hitLocation;
    color3 hitColor;
    vec3 normal;
};

__device__ inline float rfloat(uint& seed)
{
    seed ^= seed >> 16;
    seed *= 0x85ebca6b;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35;
    seed ^= seed >> 16;

    return ((float)seed / (float)UINT_MAX) * 2.0f - 1.0f;

    //uint rint = seed % 65535;
    //float random = rint / 65535.0f;
    //return random * 2.0f - 1.0f;

    //seed = seed * 747796405 + 2891336453;
    //seed = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
    //seed = (seed >> 22) ^ seed;
    //float random = seed / 4294967295.0;
    //return random * 2.0f - 1.0f;
}


struct pixelDebugInfo
{
    float fr;
    float fg;
    float fb;
    int ir;
    int ig;
    int ib;
    int ray_hits;
};