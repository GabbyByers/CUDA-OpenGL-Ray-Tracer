#pragma once

#include "dataStructures.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <string>
#include <iostream>

class InterOpRenderer
{

public:
    
    const char* vertexShaderSource = R"glsl(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
        )glsl";
    
    const char* fragmentShaderSource = R"glsl(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D uTex;
        void main() {
            FragColor = texture(uTex, TexCoord);
        }
        )glsl";

    GLFWmonitor* primary = nullptr;
    int screen_width;
    int screen_height;
    GLFWwindow* window = nullptr;
    GLuint vertShader;
    GLuint fragShader;
    GLuint shaderProgram;
    GLuint pixelBufferObject;
    cudaGraphicsResource* cudaPixelBufferObject;
    GLuint textureID;
    float quadVerts[16];
    unsigned int quadIdx[6];
    GLuint VAO, VBO, EBO;
    uchar4* devicePointer = nullptr;
    size_t size = 0;
    dim3 block;
    dim3 grid;

    double prevMouseX;
    double prevMouseY;

	InterOpRenderer(int w, int h, std::string title, bool fullScreen)
    {
        glfwInit();
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        if (fullScreen)
        {
            primary = glfwGetPrimaryMonitor();
            screen_width = glfwGetVideoMode(primary)->width;
            screen_height = glfwGetVideoMode(primary)->height;
            window = glfwCreateWindow(screen_width, screen_height, title.c_str(), primary, nullptr);
        }
        else
        {
            screen_width = w;
            screen_height = h;
            window = glfwCreateWindow(screen_width, screen_height, title.c_str(), nullptr, nullptr);
        }
        glfwMakeContextCurrent(window);
        glfwSwapInterval(0);
        gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glViewport(0, 0, screen_width, screen_height);
        vertShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertShader, 1, &vertexShaderSource, nullptr);
        glCompileShader(vertShader);
        fragShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragShader, 1, &fragmentShaderSource, nullptr);
        glCompileShader(fragShader);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertShader);
        glAttachShader(shaderProgram, fragShader);
        glLinkProgram(shaderProgram);
        glDeleteShader(vertShader);
        glDeleteShader(fragShader);
        glGenBuffers(1, &pixelBufferObject);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObject);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, screen_width * screen_height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        cudaGraphicsGLRegisterBuffer(&cudaPixelBufferObject, pixelBufferObject, cudaGraphicsMapFlagsWriteDiscard);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        quadVerts[0]  = -1; quadVerts[1]  = -1; quadVerts[2]  =  0; quadVerts[3]  =  0;
        quadVerts[4]  =  1; quadVerts[5]  = -1; quadVerts[6]  =  1; quadVerts[7]  =  0;
        quadVerts[8]  =  1; quadVerts[9]  =  1; quadVerts[10] =  1; quadVerts[11] =  1;
        quadVerts[12] = -1; quadVerts[13] =  1; quadVerts[14] =  0; quadVerts[15] =  1;
        quadIdx[0] = 0; quadIdx[1] = 1; quadIdx[2] = 2;
        quadIdx[3] = 2; quadIdx[4] = 3; quadIdx[5] = 0;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
        cudaSetDevice(0);
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");
	}

    dim3 getBlock()
    {
        return dim3(32, 32);
    }

    dim3 getGrid()
    {
        return dim3((screen_width / 32) + 1, (screen_height / 32) + 1);
    }

	~InterOpRenderer()
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glDeleteProgram(shaderProgram);
        glfwDestroyWindow(window);
        glfwTerminate();
        cudaGraphicsUnregisterResource(cudaPixelBufferObject);
        glDeleteBuffers(1, &pixelBufferObject);
        glDeleteTextures(1, &textureID);
        glDeleteVertexArrays(1, &VAO);
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        cudaDeviceReset();
        devicePointer = nullptr;
        size = 0;
	}

    void startFrame()
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        cudaGraphicsMapResources(1, &cudaPixelBufferObject, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&devicePointer, &size, cudaPixelBufferObject);
    }

    void endFrame()
    {
        cudaGraphicsUnmapResources(1, &cudaPixelBufferObject, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixelBufferObject);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_width, screen_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glUniform1i(glGetUniformLocation(shaderProgram, "uTex"), 0);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    void keyBoardControls(camera& camera)
    {
        float magic_num = 0.8f;

        vec3 forward = { camera.direction.x, 0.0f, camera.direction.z };
        forward = normalize(forward);
        vec3 up = { 0.0f, 1.0f, 0.0f };

        // Forward, Backwards
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        {
            camera.position = camera.position + forward * magic_num;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        {
            camera.position = camera.position - forward * magic_num;
        }

        // Left Right
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        {
            camera.position = camera.position - camera.right * magic_num;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        {
            camera.position = camera.position + camera.right * magic_num;
        }

        // Up Down
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        {
            camera.position = camera.position + up * magic_num;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        {
            camera.position = camera.position - up * magic_num;
        }
    }

    void mouseControls(camera& camera)
    {

        double currMouseX;
        double currMouseY;
        glfwGetCursorPos(window, &currMouseX, &currMouseY);

        double mouseRelX = currMouseX - prevMouseX;
        double mouseRelY = currMouseY - prevMouseY;

        prevMouseX = currMouseX;
        prevMouseY = currMouseY;

        ImGuiIO& io = ImGui::GetIO();
        bool mouseFree = !io.WantCaptureMouse;

        if (mouseFree)
        {
            if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
            {
                vec3 up = { 0.0f, 1.0f, 0.0f };
                float magin_num = 0.005f;

                camera.direction = quaternionRotate(camera.direction, up, magin_num * -mouseRelX);
                camera.right = quaternionRotate(camera.right, up, magin_num * -mouseRelX);
                camera.up = quaternionRotate(camera.up, up, magin_num * -mouseRelX);

                camera.direction = quaternionRotate(camera.direction, camera.right, magin_num * -mouseRelY);
                camera.up = quaternionRotate(camera.up, camera.right, magin_num * -mouseRelY);
            }
        }
    }

    void imGUI(light& light_source)
    {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::Begin("Debugger");
        ImGui::Text("Light Source");
        ImGui::SliderFloat("Light Source X Direction", &light_source.direction.x, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Source Y Direction", &light_source.direction.y, -1.0f, 1.0f);
        ImGui::SliderFloat("Light Source Z Direction", &light_source.direction.z, -1.0f, 1.0f);
        ImGui::End();
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
};

