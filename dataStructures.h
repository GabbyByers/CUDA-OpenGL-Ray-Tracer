#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <cstdlib>

using uchar = unsigned char;

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

//struct color
//{
//    float r;
//    float g;
//    float b;
//    float a;
//
//    __host__ __device__ color operator * (const color& other) const // color multiplication
//    {
//        return { r * r, g * g, b * b, a * a };
//    }
//};

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
    uchar4 color;
};

struct light
{
    vec3 direction;
    uchar4 color;
};

struct plane
{
    vec3 position;
    vec3 normal;
};






struct quat
{
    float w;
    float x;
    float y;
    float z;

    quat operator * (const quat& b) const // quaternion multiplication
    {
        return
        {
            (w * b.w) - (x * b.x) - (y * b.y) - (z * b.z),
            (w * b.x) + (x * b.w) + (y * b.z) - (z * b.y),
            (w * b.y) - (x * b.z) + (y * b.w) + (z * b.x),
            (w * b.z) + (x * b.y) - (y * b.x) + (z * b.w)
        };
    }
};

inline vec3 quaternionRotate(vec3& a, vec3& b, float theta)
{
    float half = theta / 2.0f;
    float s = sin(half);

    quat q = { cos(half), b.x * s, b.y * s, b.z * s };
    quat qinv = { q.w, -q.x, -q.y, -q.z };
    quat p = { 0.0f, a.x, a.y, a.z };
    
    quat qp = q * p;
    quat result = qp * qinv;

    return { result.x, result.y, result.z };
}

