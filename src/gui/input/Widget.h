#ifndef WIDGET_H
#define WIDGET_H

#include "../include/imgui/imgui.h"
#include "../include/imgui/backends/imgui_impl_glfw.h"
#include "../include/imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "linalg/linalg.h"

class Widget {
public:
  Point3 cameraDir;
  Vec3 cameraPos;
  Point3 lightPos;
  Color3 bgColor; // TODO: widget

  bool paused;
  bool renderOnce;
  char* fps;

  int tfComboSelected;
  int opacityK;
  float opacityKReal;
  float sigmoidShift;
  float sigmoidExp;

  ImGuiIO io;

  void tick(double fps);
  void render();
  void copyToDevice();

  Widget(GLFWwindow* window);
  ~Widget();
};



#endif // WIDGET_H
