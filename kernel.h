#pragma once
#include "dataStructures.h"
#include "InterOpRenderer.h"

__global__ inline void kernel(uchar4* devPtr, int width, int height, camera camera, sphere* spheres, int numSpheres, light light_source, plane floor_plane)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width) || (y >= height)) return;
    int index = y * width + x;

    float u = ((2.0f * ((float)x / width)) - 1.0f) * ((float)width / height);
    float v = (2.0f * ((float)y / height)) - 1.0f;

    vec3 rayDir = normalize((camera.direction * camera.depth) + (camera.up * v) + (camera.right * u));
    
    uchar r;
    uchar g;
    uchar b;
    uchar a;
    
    // Collide with Spheres
    float closest_t = FLT_MAX;
    int closest_sphere = 0;
    for (int i = 0; i < numSpheres; i++)
    {
        vec3 V = camera.position - spheres[i].position;
    
        float A = (rayDir.x * rayDir.x) + (rayDir.y * rayDir.y) + (rayDir.z * rayDir.z);
        float B = 2 * ((V.x * rayDir.x) + (V.y * rayDir.y) + (V.z * rayDir.z));
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
            closest_t = t;
            closest_sphere = i;
        }
    }
    
    // Collide with Floor
    float plane_t = dot((floor_plane.position - camera.position), floor_plane.normal) / dot(rayDir, floor_plane.normal);
    if (plane_t <= 0)
        plane_t = FLT_MAX;
    
    // Draw Floor
    if (plane_t < closest_t)
    {
        uchar4 floor_color = make_uchar4(0, 0, 0, 255);
        uchar4 light_gray = make_uchar4(150, 150, 150, 255);
        uchar4 dark_gray = make_uchar4(80, 80, 80, 255);
    
        vec3 hit = camera.position + (rayDir * plane_t);
        if (hit.z < 0)
            hit.z -= 1.0f;
        if (hit.x < 0)
            hit.x -= 1.0f;
    
        if ((int)hit.x % 2 == 0)
        {
            if ((int)hit.z % 2 == 0)
            {
                floor_color = dark_gray;
            }
            else
            {
                floor_color = light_gray;
            }
        }
        else
        {
            if ((int)hit.z % 2 == 0)
            {
                floor_color = light_gray;
            }
            else
            {
                floor_color = dark_gray;
            }
        }
    
        devPtr[index] = floor_color;
        return;
    }
    
    // Draw Sphere
    if (closest_t < plane_t)
    {
        vec3 sphere_normal = normalize((camera.position + rayDir * closest_t) - spheres[closest_sphere].position);
        vec3 reflected_rayDir = rayDir - (sphere_normal * (2 * dot(rayDir, sphere_normal)));
    
        float light_intensity = dot(reflected_rayDir, light_source.direction);
        //float light_intensity = dot(sphere_normal, light_source.direction);
    
        if (light_intensity > 0.0f)
        {
            devPtr[index] = spheres[closest_sphere].color;
            return;
        }
    
        devPtr[index] = make_uchar4(0, 0, 0, 255);
        return;
    }
    
    // Draw Sky
    r = 31;
    g = 165;
    b = 255;
    a = 255;
    devPtr[index] = make_uchar4(r, g, b, a);
    return;
}