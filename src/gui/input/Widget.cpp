#include "Widget.h"
#include "linalg/linalg.h"
#include "consts.h"


Widget::Widget(GLFWwindow* window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  this->io = ImGui::GetIO();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init();

  this->cameraDir = Vec3::init(0.4, 0.6, 1.0).normalize();
  this->cameraPos = Point3::init(-0.7, -1.0, -2.0);
  this->cameraUp = Vec3::init(0.0, 1.0, 0.0).normalize();
  this->lightPos = Point3::init(1.5, 2.0, -1.0);
  this->paused = true;
  this->renderOnce = false;
};

void Widget::tick() {
  if (this->renderOnce) {
    this->renderOnce = false;
    this->paused = true;
  }

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  
  // input widgets
  float min = -1, max = 1;

  ImGui::Begin("Light Controls");
  ImGui::DragScalar("X coordinate", ImGuiDataType_Double, &this->lightPos.x, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Y coordinate", ImGuiDataType_Double, &this->lightPos.y, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Z coordinate", ImGuiDataType_Double, &this->lightPos.z, 0.005f, &min, &max, "%.3f");
  ImGui::End();

  ImGui::Begin("Pause");
  if (ImGui::Button(this->paused ? "Unpause" : "Pause")) this->paused = !this->paused;
  if (ImGui::Button("render Once")) {
    this->paused = !this->paused;
    this->renderOnce = true;
  }
  ImGui::End();

  ImGui::Begin("Camera Controls");
  ImGui::DragScalar("X position", ImGuiDataType_Double, &this->cameraPos.x, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Y position", ImGuiDataType_Double, &this->cameraPos.y, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Z position", ImGuiDataType_Double, &this->cameraPos.z, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("X direction", ImGuiDataType_Double, &this->cameraDir.x, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Y direction", ImGuiDataType_Double, &this->cameraDir.y, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Z direction", ImGuiDataType_Double, &this->cameraDir.z, 0.005f, &min, &max, "%.3f");
  ImGui::End();
  
  copyToDevice();
}

void Widget::render() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Widget::copyToDevice() {
  cudaMemcpyToSymbol(&d_cameraPos, &this->cameraPos, sizeof(Point3));
  cudaMemcpyToSymbol(&d_cameraDir, &this->cameraDir, sizeof(Vec3));
  cudaMemcpyToSymbol(&d_cameraUp, &this->cameraUp, sizeof(Vec3));
  cudaMemcpyToSymbol(&d_lightPos, &this->lightPos, sizeof(Point3));
}

Widget::~Widget() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}
