#include "kernel.h"

int main()
{
    InterOpRenderer iOpRenderer(1200, 700, "CUDA-OpenGL Ray-Tracer by Gabby Byers 2025", false);
    dim3 block = iOpRenderer.getBlock();
    dim3 grid = iOpRenderer.getGrid();

    // Camera
    camera camera;
    camera.position =  { 15.0f, 0.0f, 2.0f };
    camera.direction = { -1.0f, 0.0f, 0.0f };
    camera.up =        { 0.0f, 1.0f, 0.0f };
    camera.right =     { 0.0f, 0.0f, -1.0f };
    camera.depth =       1.5f;

    // Spheres
    int numSpheres = 6;
    sphere* cpuSpheres = new sphere[numSpheres];

    cpuSpheres[0] = sphere{ { 2.0,   0.0f, -2.0 }, 1.5f, {1.0f, 1.0f, 0.1f}, false };
    cpuSpheres[1] = sphere{ { 5.0,   0.0f, -4.0 }, 1.0f, {0.1f, 1.0f, 1.0f}, false };
    cpuSpheres[2] = sphere{ { 9.0,   0.0f, -4.0 }, 2.0f, {1.0f, 1.0f, 1.0f}, false };
    cpuSpheres[3] = sphere{ { 12.0,  0.0f, -1.0 }, 1.0f, {0.1f, 1.0f, 0.1f}, false };
    cpuSpheres[4] = sphere{ { 6.0,   1.0f, 3.0  }, 2.5f, {1.0f, 1.0f, 1.0f}, true };
    cpuSpheres[5] = sphere{ { 6.0, -100.0f, 0.0  }, 99.0f, {0.5f, 0.5f, 1.0f}, false };
    
    sphere* devSpheres = nullptr;
    cudaMalloc((void**)&devSpheres, sizeof(sphere) * numSpheres);
    cudaMemcpy(devSpheres, cpuSpheres, sizeof(sphere) * numSpheres, cudaMemcpyHostToDevice);

    // Controls
    int maxBounceLimit = 5;
    int raysPerPixel = 1;
    float emission = 1.0f;

    // Debug Capture
    pixelDebugInfo* hostDebugCapture = new pixelDebugInfo;
    pixelDebugInfo* deviceDebugCapture = nullptr;
    cudaMalloc((void**)&deviceDebugCapture, sizeof(pixelDebugInfo));


    while (!glfwWindowShouldClose(iOpRenderer.window))
    {
        iOpRenderer.startFrame();

        int mouse_x = iOpRenderer.prevMouseX;
        int mouse_y = iOpRenderer.prevMouseY;

        uchar4* devPtr = iOpRenderer.devicePointer;
        int width = iOpRenderer.screen_width;
        int height = iOpRenderer.screen_height;
        kernel <<<grid, block>>> (devPtr, width, height, camera, devSpheres, numSpheres, maxBounceLimit, raysPerPixel, emission, deviceDebugCapture, mouse_x, mouse_y);
        cudaDeviceSynchronize();
        
        cudaMemcpy(hostDebugCapture, deviceDebugCapture, sizeof(pixelDebugInfo), cudaMemcpyDeviceToHost);

        iOpRenderer.endFrame();
        iOpRenderer.keyBoardControls(camera);
        iOpRenderer.mouseControls(camera);
        iOpRenderer.imGUI(maxBounceLimit, raysPerPixel, emission, hostDebugCapture);
    }

    delete[] cpuSpheres;
    cudaFree(devSpheres);

    return 0;
}

