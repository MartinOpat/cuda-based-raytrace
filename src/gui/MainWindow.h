#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "Quad.h"
#include "Shader.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include "input/Widget.h"
#include <fstream>
#include <iostream>


class Window {
public:
  unsigned int w;
  unsigned int h;
  float* data; 

  Window(unsigned int w, unsigned int h);

  int init(float* data);
  void free(float* data);
  void resize(unsigned int w, unsigned int h);

private:
  GLFWwindow* window;
  std::unique_ptr<Quad> quad;
  Widget* widget;

  std::ofstream gpuPerf;
  std::ofstream cpuPerf;

	std::chrono::steady_clock::time_point last_frame;

  void tick();
  int init_quad(float* data);

  std::unique_ptr<Shader> shader;
};
#endif // MAINWINDOW_H
