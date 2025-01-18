#include "Quad.h"

#include "cuda_error.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_gl_interop.h>
#include <iostream>

Quad::Quad(unsigned int w, unsigned int h) {
  this->w = w;
  this->h = h;

  std::vector<float> vertices = {
    -1.0f,  1.0f,  0.0f, 1.0f,
    -1.0f, -1.0f,  0.0f, 0.0f,
     1.0f, -1.0f,  1.0f, 0.0f,

    -1.0f,  1.0f,  0.0f, 1.0f,
     1.0f, -1.0f,  1.0f, 0.0f,
     1.0f,  1.0f,  1.0f, 1.0f
  };

  glGenBuffers(1, &VBO);
  glGenVertexArrays(1, &VAO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  
  // copy vertex data to buffer on gpu
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  // set our vertex attributes pointers
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // texture stuff
  glGenBuffers(1, &PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4, NULL, GL_DYNAMIC_COPY);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  glEnable(GL_TEXTURE_2D);

  glGenTextures(1, &tex);

  glBindTexture(GL_TEXTURE_2D, tex);

  // parameters for texture
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

};

void Quad::make_fbo(){
  glGenFramebuffers(1, &fb);
  glBindFramebuffer(GL_FRAMEBUFFER, fb);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Quad::~Quad() {
  check_cuda_errors(cudaGraphicsUnregisterResource(CGR));
};


void Quad::cuda_init(float* data) {
  check_cuda_errors(cudaGraphicsGLRegisterBuffer(&this->CGR, this->PBO, cudaGraphicsRegisterFlagsNone));
  this->renderer = std::make_unique<Raycaster>(this->CGR, this->w, this->h, data);
};


void Quad::render() {
  check_cuda_errors(cudaGetLastError());
  glBindTexture(GL_TEXTURE_2D, 0);
  this->renderer->render(); 
  glBindTexture(GL_TEXTURE_2D, this->tex);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->PBO);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->w, this->h, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
};


void Quad::resize(unsigned int w, unsigned int h) {
  this->w = w;
  this->h = h;

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * 4, NULL, GL_DYNAMIC_COPY);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  glBindTexture(GL_TEXTURE_2D, this->tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  glBindFramebuffer(GL_FRAMEBUFFER, this->fb);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, this->tex, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);    

  if (this->renderer != nullptr) {
    check_cuda_errors(cudaGraphicsUnregisterResource(CGR));
    check_cuda_errors(cudaGraphicsGLRegisterBuffer(&this->CGR, this->PBO, cudaGraphicsRegisterFlagsNone));

    this->renderer->resources = this->CGR;
    this->renderer->resize(w, h);
  }
};
