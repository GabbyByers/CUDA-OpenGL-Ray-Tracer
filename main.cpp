#include "kernel.h"

int main()
{
    InterOpRenderer iOpRenderer(1000, 500, "CUDA-OpenGL Ray-Tracer by Gabby Byers 2025", false);
    dim3 block = iOpRenderer.getBlock();
    dim3 grid = iOpRenderer.getGrid();

    // Camera
    camera camera;
    camera.position =  { 0.0f, 10.0f, 0.0f };
    camera.direction = { 1.0f, 0.0f, 0.0f };
    camera.up =        { 0.0f, 1.0f, 0.0f };
    camera.right =     { 0.0f, 0.0f, 1.0f };
    camera.depth =       1.5f;

    // Spheres
    int numSpheres = 100;
    sphere* cpuSpheres = new sphere[numSpheres];
    for (int i = 0; i < numSpheres; i++)
    {
        sphere s;
        s.position = { randf(15.0f, 25.0f), randf(5.0f, 15.0f), randf(-10.0f, 10.0f) };
        s.radius = randf(0.2f, 0.8f);
        s.color = make_uchar4(rand() % 255, rand() % 255, rand() % 255, 255);
        cpuSpheres[i] = s;
    }
    sphere* devSpheres = nullptr;
    cudaMalloc((void**)&devSpheres, sizeof(sphere) * numSpheres);
    cudaMemcpy(devSpheres, cpuSpheres, sizeof(sphere) * numSpheres, cudaMemcpyHostToDevice);

    // Light Source
    light light_source;
    light_source.direction = normalize(vec3{ 1.0f, 1.0f, 1.0f });
    light_source.color = make_uchar4(255, 255, 255, 255);

    // Floor Plane
    plane floor_plane;
    floor_plane.position = vec3{ 0.0f, 0.0f, 0.0f };
    floor_plane.normal = vec3{ 0.0f, 1.0f, 0.0f };

    while (!glfwWindowShouldClose(iOpRenderer.window))
    {
        iOpRenderer.startFrame();

        uchar4* devPtr = iOpRenderer.devicePointer;
        int width = iOpRenderer.screen_width;
        int height = iOpRenderer.screen_height;
        kernel <<<grid, block>>> (devPtr, width, height, camera, devSpheres, numSpheres, light_source, floor_plane);
        cudaDeviceSynchronize();
        
        iOpRenderer.endFrame();
        iOpRenderer.keyBoardControls(camera);
        iOpRenderer.mouseControls(camera);
        iOpRenderer.imGUI(light_source);


        light_source.direction = normalize(light_source.direction);
    }

    delete[] cpuSpheres;
    cudaFree(devSpheres);

    return 0;
}

