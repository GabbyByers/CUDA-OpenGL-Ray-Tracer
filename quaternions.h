#pragma once

#include "dataStructures.h"

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