#pragma once
#include "dataStructures.h"
#include "InterOpRenderer.h"
#include <cmath>

__device__ inline hitInfo sphereCollision(sphere* spheres, int numSpheres, vec3 ray_origin, vec3 ray_direction)
{
    float closest_t = FLT_MAX;
    int index_closest_sphere = -1;
    color3 curr_color;
    bool didHit = false;

    for (int i = 0; i < numSpheres; i++)
    {
        vec3 V = ray_origin - spheres[i].position;

        float A = (ray_direction.x * ray_direction.x) + (ray_direction.y * ray_direction.y) + (ray_direction.z * ray_direction.z);
        float B = 2 * ((V.x * ray_direction.x) + (V.y * ray_direction.y) + (V.z * ray_direction.z));
        float C = (V.x * V.x) + (V.y * V.y) + (V.z * V.z) - (spheres[i].radius * spheres[i].radius);
        float discriminant = (B * B) - (4.0f * A * C);

        if (discriminant <= 0.0f)
            continue;

        float t1 = ((-1.0f * B) + sqrt(discriminant)) / (2.0f * A);
        float t2 = ((-1.0f * B) - sqrt(discriminant)) / (2.0f * A);
        float t = fminf(t1, t2);

        if (t <= 0.0f)
            continue;

        if (t < closest_t)
        {
            didHit = true;
            closest_t = t;
            index_closest_sphere = i;
            curr_color = spheres[i].color;
        }
    }

    if (didHit)
    {
        vec3 hitLocation = ray_origin + ray_direction * closest_t;
        vec3 normal = normalize(hitLocation - spheres[index_closest_sphere].position);

        return hitInfo
        {
            true,
            spheres[index_closest_sphere].isLightSource,
            hitLocation,
            curr_color,
            normal
        };
    }

    return hitInfo
    {
        false
    };
    
}

__device__ inline vec3 randomHemisphereDirection(vec3 normal, uint& seed)
{
    int search_limit = 1;
 
    float rx;
    float ry;
    float rz;

    for (int i = 0; i < search_limit; i++)
    {
        rx = rfloat(seed);
        ry = rfloat(seed);
        rz = rfloat(seed);

        if ((rx * rx + ry * ry + rz * rz) < 1.0f)
        {
            break;
        }
    }

    vec3 result = normalize(vec3{ rx, ry, rz });

    if (dot(result, normal) < 0.0f)
    {
        result.x = result.x * -1.0f;
        result.y = result.y * -1.0f;
        result.z = result.z * -1.0f;
    }

    return result;
}

__global__ inline void kernel(uchar4* devPtr, int width, int height, camera camera, sphere* spheres, int numSpheres, int maxBounceLimit, int raysPerPixel, float emission, pixelDebugInfo* deviceDebugCapture, int mouse_x, int mouse_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width) || (y >= height)) return;
    int index = y * width + x;
    uint seed = index;

    float u = ((2.0f * ((float)x / width)) - 1.0f) * ((float)width / height);
    float v = (2.0f * ((float)y / height)) - 1.0f;

    color3 ray_color = { 1.0f, 1.0f, 1.0f };
    color3 incoming_light = { 0.0f, 0.0f, 0.0f };

    int hits = 0;

    for (int i = 0; i < raysPerPixel; i++)
    {
        vec3 ray_origin = camera.position;
        vec3 ray_direction = normalize((camera.direction * camera.depth) + (camera.up * v) + (camera.right * u));

        for (int j = 0; j < maxBounceLimit; j++)
        {
            hitInfo info = sphereCollision(spheres, numSpheres, ray_origin, ray_direction);

            if (info.didHit)
            {
                if (info.didHitLight)
                {
                    incoming_light += (info.hitColor * ray_color) * emission;
                    hits++;
                    break;
                }
                ray_origin = info.hitLocation;
                ray_direction = randomHemisphereDirection(info.normal, seed);
                ray_color *= info.hitColor;
            }
            else
            {
                break;
            }
        }
    }

    //incoming_light /= raysPerPixel;
    //
    //float max_channel = fmax(incoming_light.r, fmax(incoming_light.g, incoming_light.b));
    //if (max_channel > 1.0f)
    //    incoming_light /= max_channel;
    //
    //incoming_light *= 255.0f;

    uchar r = (uchar)fminf(incoming_light.r * 255.0f, 255.0f);
    uchar g = (uchar)fminf(incoming_light.g * 255.0f, 255.0f);
    uchar b = (uchar)fminf(incoming_light.b * 255.0f, 255.0f);

    if (x == mouse_x)
    {
        if (y == height - mouse_y)
        {
            deviceDebugCapture->fr = incoming_light.r;
            deviceDebugCapture->fg = incoming_light.g;
            deviceDebugCapture->fb = incoming_light.b;
            deviceDebugCapture->ir = r;
            deviceDebugCapture->ig = g;
            deviceDebugCapture->ib = b;
            deviceDebugCapture->ray_hits = hits;
            uchar r = 255;
            uchar g = 0;
            uchar b = 0;
            devPtr[index] = make_uchar4(r, g, b, 255);
            return;
        }
    }

    devPtr[index] = make_uchar4(r, g, b, 255);
    return;
}