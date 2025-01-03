#ifndef QUAD_H
#define QUAD_H

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "illumination/Raycaster.h"

#include <vector>
#include <memory>

class Quad {
public:
  unsigned int VAO;
  unsigned int VBO;
  unsigned int PBO;
  cudaGraphicsResource_t CGR;

  unsigned int tex;
  unsigned int fb;

  unsigned int w;
  unsigned int h;

  std::unique_ptr<Raycaster> renderer; 

  Quad(unsigned int w, unsigned int h);
  ~Quad(); 

  void render();
  void resize(unsigned int w, unsigned int h);
  void cuda_init();

};
#endif // QUAD_H
