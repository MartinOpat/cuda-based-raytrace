#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "Quad.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>


class Window {
public:
  unsigned int w;
  unsigned int h;

  Window(unsigned int w, unsigned int h);

  int init();
  void free();
  void resize(unsigned int w, unsigned int h);

private:
  GLFWwindow* window;
  std::unique_ptr<Quad> current_quad;

	std::chrono::steady_clock::time_point last_frame;

  void tick();
  int init_quad();

  // std::unique_ptr<Shader> shader;
};
#endif // MAINWINDOW_H
