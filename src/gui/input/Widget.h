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
  double pitch, yaw, roll;
  Vec3 cameraUp;

  // simulation cotnrols
  bool renderOnce;
  char* fps;
  ImGuiIO io;
  char *dateString;
  int samplesPerPixel;

  // transfer function controls
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
  float specularStrength;
  int shininess;

  // miscellaneous
  Point3 lightPos;
  Color3 bgColor; // TODO: widget
  
  void copyToDevice();
  void resetCamera();
  void resetLight();

};



#endif // WIDGET_H
