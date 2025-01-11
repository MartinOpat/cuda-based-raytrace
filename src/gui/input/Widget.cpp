#include "Widget.h"
#include "linalg/linalg.h"
#include "consts.h"
#include <cstdio>
#include <stdlib.h>


Widget::Widget(GLFWwindow* window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  this->io = ImGui::GetIO();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init();

  this->cameraPos = Point3::init(50.0f, -50.0f, -75.0f);  // Camera for partially trimmed data set
  // this->cameraPos = Point3::init(300.0f, 200.0f, -700.0f);  // Camera for full data set
  
  Vec3 h_center = Vec3::init((float)VOLUME_WIDTH/2.0f, (float)VOLUME_HEIGHT/2.0f, (float)VOLUME_DEPTH/2.0f);
  this->cameraDir = (h_center - this->cameraPos).normalize();

  this->bgColor = Color3::init(0.1f, 0.1f, 0.1f);
  this->lightPos = Point3::init(1.5, 2.0, -1.0);

  this->fps = (char*)malloc(512*sizeof(char));
  this->paused = true;
  this->renderOnce = false;

  this->opacityK = 0;
  this->sigmoidOne = 0.5f;
  this->sigmoidTwo = -250.0f;
};

// TODO: can be marginally improvement by only copying changed values to device - however we're dealing with individual floats here so i dont think the benefit would be all that obvious.
void Widget::tick(double fps) {
  if (this->renderOnce) {
    this->renderOnce = false;
    this->paused = true;
  }

  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  
  // input widgets
  float min = -1, max = 1;

  ImGui::Begin("Transfer Function Controls");
  ImGui::DragInt("k (log [1e-10, 1])", &this->opacityK, 1, 0, 100, "%d%%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::DragFloat("sigmoidOne", &this->sigmoidOne, 0.01f, 0.0f, 1.0f, "%.2f");
  ImGui::InputFloat("sigmoidTwo", &this->sigmoidTwo, 10.0f, 100.0f, "%.0f");
  ImGui::End();

  ImGui::Begin("Light Controls");
  ImGui::DragScalar("X coordinate", ImGuiDataType_Double, &this->lightPos.x, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Y coordinate", ImGuiDataType_Double, &this->lightPos.z, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Z coordinate", ImGuiDataType_Double, &this->lightPos.y, 0.5f, &min, &max, "%.3f");
  ImGui::End();

  ImGui::Begin("Miscellaneous");
  if (ImGui::Button(this->paused ? "Unpause" : "Pause")) this->paused = !this->paused;
  ImGui::SameLine();
  if (ImGui::Button("Render once")) {
    this->paused = !this->paused;
    this->renderOnce = true;
  }
  sprintf(this->fps, "%.3f fps\n", fps);
  ImGui::Text(this->fps);
  ImGui::End();

  ImGui::Begin("Camera Controls");
  ImGui::DragScalar("X position", ImGuiDataType_Double, &this->cameraPos.x, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Y position", ImGuiDataType_Double, &this->cameraPos.z, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Z position", ImGuiDataType_Double, &this->cameraPos.y, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("X direction", ImGuiDataType_Double, &this->cameraDir.x, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Y direction", ImGuiDataType_Double, &this->cameraDir.z, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Z direction", ImGuiDataType_Double, &this->cameraDir.y, 0.005f, &min, &max, "%.3f");
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
  cudaMemcpyToSymbol(&d_lightPos, &this->lightPos, sizeof(Point3));
  cudaMemcpyToSymbol(&d_backgroundColor, &this->bgColor, sizeof(Color3));
  
  // cudaMemcpyToSymbol(&d_opacityK, &this->opacityK, sizeof(float));
  this->opacityKReal = std::pow(10.0f, (-10 + 0.1 * this->opacityK));
  cudaMemcpyToSymbol(&d_opacityK, &this->opacityKReal, sizeof(float));

  cudaMemcpyToSymbol(&d_sigmoidOne, &this->sigmoidOne, sizeof(float));
  cudaMemcpyToSymbol(&d_sigmoidTwo, &this->sigmoidTwo, sizeof(float));
}

Widget::~Widget() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  free(this->fps);
}
