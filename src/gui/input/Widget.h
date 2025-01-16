#ifndef WIDGET_H
#define WIDGET_H

#include "../include/imgui/imgui.h"
#include "../include/imgui/backends/imgui_impl_glfw.h"
#include "../include/imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <string>
#include "linalg/linalg.h"

class Widget {
public:
  bool paused;
  bool dateChanged;
  int date;

  void tick(double fps);
  void render();

  Widget(GLFWwindow* window);
  ~Widget();

private:
  // camera controls
  Point3 cameraDir;
  Vec3 cameraPos;

  // simulation cotnrols
  bool renderOnce;
  char* fps;
  ImGuiIO io;
  char *dateString;

  // transfer function controls
  int tfComboSelected;
  int opacityK;
  float opacityKReal;
  float sigmoidOne;
  float sigmoidTwo;

  // miscellaneous
  Point3 lightPos;
  Color3 bgColor; // TODO: widget
  
  void copyToDevice();
  void resetCamera();
  void resetLight();

};



#endif // WIDGET_H
