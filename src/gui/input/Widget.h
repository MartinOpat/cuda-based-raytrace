#ifndef WIDGET_H
#define WIDGET_H

#include "../include/imgui/imgui.h"
#include "../include/imgui/backends/imgui_impl_glfw.h"
#include "../include/imgui/backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include "linalg/linalg.h"

class Widget {
public:
  double pitch, yaw, roll;
  Vec3 cameraUp;
  Vec3 cameraDir;
  Point3 cameraPos;
  Point3 lightPos;
  Color3 bgColor; // TODO: widget

  bool paused;
  bool renderOnce;
  char* fps;
  int samplesPerPixel;

  int tfComboSelected;
  int tfComboSelectedColor;
  int opacityK;
  float opacityKReal;
  float sigmoidShift;
  float sigmoidExp;
  float alphaAcumLimit;
  int opacityConst;
  float opacityConstReal;
  bool showSilhouettes;
  float silhouettesThreshold;
  float levoyFocus;
  float levoyWidth;

  ImGuiIO io;

  void tick(double fps);
  void render();
  void copyToDevice();

  Widget(GLFWwindow* window);
  ~Widget();
};



#endif // WIDGET_H
