#include "Widget.h"
#include "linalg/linalg.h"
#include "consts.h"
#include <cstdio>
#include <stdlib.h>
#include <iostream>


Widget::Widget(GLFWwindow* window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  this->io = ImGui::GetIO();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init();

  // this->cameraPos = Point3::init(64.0f, 42.5f, 250.0f);  // Camera for partially trimmed data set
  this->cameraPos = Point3::init(59.0f, 77.5f, -18.0f);  // Camera for partially trimmed data set
  // this->cameraPos = Point3::init(300.0f, 200.0f, -700.0f);  // Camera for full data set
  // this->pitch = -1.7;
  // this->yaw = 1.475;
  // this->roll = 0;
  this->pitch = 0.7;
  this->yaw = 4.85;
  this->roll = -0.010;
  this->cameraDir = Vec3::getDirectionFromEuler(pitch, yaw, roll);
  // this->cameraDir = Point3::init(0.074f, -0.301f, -2.810f);
  
  // Vec3 h_center = Vec3::init((float)VOLUME_WIDTH/2.0f, (float)VOLUME_HEIGHT/2.0f, (float)VOLUME_DEPTH/2.0f);
  // this->cameraDir = (h_center - this->cameraPos).normalize();

  // Vec3 h_center = Vec3::init((float)VOLUME_WIDTH, (float)VOLUME_HEIGHT, (float)VOLUME_DEPTH);
  // this->cameraDir = (h_center - this->cameraPos).normalize();

  this->bgColor = Color3::init(0.1f, 0.1f, 0.1f);
  this->lightPos = Point3::init(1.5, 2.0, -1.0);

  this->fps = (char*)malloc(512*sizeof(char));
  this->paused = true;
  this->renderOnce = false;
  this->samplesPerPixel = 1;

  this->opacityK = 0;
  this->sigmoidShift = 0.5f;
  this->sigmoidExp = -250.0f;
  this->alphaAcumLimit = 0.4f;
  this->tfComboSelected = 2;
  this->tfComboSelectedColor = 0;
  this->opacityConst = 100;
  this->showSilhouettes = false;
  this->silhouettesThreshold = 0.02f;
  this->levoyFocus = 0.5;
  this->levoyWidth = 1;
};

// REFACTOR: should probably not have all the logic in one function; something like a list of ImplementedWidgets with each a Render() function (a la interface) would be better.
// TODO: can be marginally improved by only copying changed values to device - however we're dealing with individual floats here so i dont think the benefit would be all that obvious.
// TODO: wrap basically all ImGui calls in if statements; better form + allows for checking return values / errors.
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
  ImGui::DragInt("Grad. exp. (log [1e-10, 1])", &this->opacityK, 1, 0, 100, "%d%%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::DragFloat("Sig. shift", &this->sigmoidShift, 0.01f, 0.0f, 1.0f, "%.2f");
  ImGui::InputFloat("Sig. sxp", &this->sigmoidExp, 10.0f, 100.0f, "%.0f");
  ImGui::DragFloat("Alpha accum. limit", &this->alphaAcumLimit, 0.01f, 0.0f, 1.0f, "%.2f");
  ImGui::DragInt("Opacity const. (log [1e-5, 1])", &this->opacityConst, 1, 0, 100, "%d%%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::DragFloat("Levoy Width", &this->levoyWidth, 0.01f, 0.0f, 100.0f, "%.2f");
  // ImGui::DragFloat("Levoy Focus", &this->levoyFocus, 0.01f, 250.0f, 350.0f, "%.2f");
  ImGui::DragFloat("Levoy Focus", &this->levoyFocus, 0.01f, 0.0f, 1.0f, "%.2f");
  
  // the items[] contains the entries for the combobox. The selected index is stored as an int on this->tfComboSelected
  // the default entry is set in the constructor, so if you want that to be a specific entry just change it
  // whatever value is selected here is available on the gpu as d_tfComboSelected.
  const char* items[] = {"Opacity - gradient", "Opacity - sigmoid", "Opacity - constant", "Opacity - levoy"};
  if (ImGui::BeginCombo("Transfer function", items[this->tfComboSelected])) {
    // std::cout << "hello???\n";
    for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
      // std::cout << "letsssssa a asdfa???\n";
      const bool is_selected = (this->tfComboSelected == n);
      if (ImGui::Selectable(items[n], is_selected))
        this->tfComboSelected = n;
      if (is_selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  // Same comments as above apply
  const char* items2[] = {"Python-like", "BPR", "Greyscale", "..."};
  if (ImGui::BeginCombo("Color map", items2[this->tfComboSelectedColor])) {
    for (int n = 0; n < IM_ARRAYSIZE(items2); n++) {
      const bool is_selected = (this->tfComboSelectedColor == n);
      if (ImGui::Selectable(items2[n], is_selected))
        this->tfComboSelectedColor = n;
      if (is_selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }

  if (ImGui::Button(this->showSilhouettes ? "Hide Silhouettes" : "Show Silhouettes")) this->showSilhouettes = !this->showSilhouettes;
  ImGui::DragFloat("Silhouettes threshold", &this->silhouettesThreshold, 0.001f, 0.0f, 0.5f, "%.3f");
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
  ImGui::Text("%s", this->fps);
  ImGui::DragInt("Samples per pixel", &this->samplesPerPixel, 1, 1, 16, "%d", ImGuiSliderFlags_AlwaysClamp);
  ImGui::End();

  ImGui::Begin("Camera Controls");
  ImGui::DragScalar("X position", ImGuiDataType_Double, &this->cameraPos.x, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Y position", ImGuiDataType_Double, &this->cameraPos.z, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Z position", ImGuiDataType_Double, &this->cameraPos.y, 0.5f, &min, &max, "%.3f");
  // ImGui::DragScalar("X direction", ImGuiDataType_Double, &this->cameraDir.x, 0.005f, &min, &max, "%.3f");
  // ImGui::DragScalar("Y direction", ImGuiDataType_Double, &this->cameraDir.z, 0.005f, &min, &max, "%.3f");
  // ImGui::DragScalar("Z direction", ImGuiDataType_Double, &this->cameraDir.y, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Pitch", ImGuiDataType_Double, &this->pitch, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Yaw", ImGuiDataType_Double, &this->yaw, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Roll", ImGuiDataType_Double, &this->roll, 0.005f, &min, &max, "%.3f");
  ImGui::End();

  this->cameraDir.setDirectionFromEuler(pitch, yaw, roll);

  // Calculate upCamera
  Vec3 arbitraryVector = Vec3::init(1, 0, 0);
  cameraUp = arbitraryVector.cross(cameraDir);
  cameraUp.normalize();
  cameraUp.rotateAroundAxis(cameraDir, roll);

  copyToDevice();
}

void Widget::render() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Widget::copyToDevice() {
  cudaMemcpyToSymbol(&d_cameraPos, &this->cameraPos, sizeof(Point3));
  cudaMemcpyToSymbol(&d_cameraUp, &this->cameraUp, sizeof(Point3));
  cudaMemcpyToSymbol(&d_cameraDir, &this->cameraDir, sizeof(Vec3));
  cudaMemcpyToSymbol(&d_lightPos, &this->lightPos, sizeof(Point3));
  cudaMemcpyToSymbol(&d_backgroundColor, &this->bgColor, sizeof(Color3));

  cudaMemcpyToSymbol(&d_samplesPerPixel, &this->samplesPerPixel, sizeof(int));
  
  // cudaMemcpyToSymbol(&d_opacityK, &this->opacityK, sizeof(float));
  this->opacityKReal = std::pow(10.0f, (-10 + 0.1 * this->opacityK));
  cudaMemcpyToSymbol(&d_opacityK, &this->opacityKReal, sizeof(float));

  cudaMemcpyToSymbol(&d_sigmoidShift, &this->sigmoidShift, sizeof(float));
  cudaMemcpyToSymbol(&d_sigmoidExp, &this->sigmoidExp, sizeof(float));
  cudaMemcpyToSymbol(&d_alphaAcumLimit, &this->alphaAcumLimit, sizeof(float));
  cudaMemcpyToSymbol(&d_tfComboSelected, &this->tfComboSelected, sizeof(int));
  cudaMemcpyToSymbol(&d_showSilhouettes, &this->showSilhouettes, sizeof(bool));
  cudaMemcpyToSymbol(&d_silhouettesThreshold, &this->silhouettesThreshold, sizeof(float));
  cudaMemcpyToSymbol(&d_levoyFocus, &this->levoyFocus, sizeof(float));
  cudaMemcpyToSymbol(&d_levoyWidth, &this->levoyWidth, sizeof(float));

  this->opacityConstReal = std::pow(10.0f, (-5 + 0.05 * this->opacityConst));
  cudaMemcpyToSymbol(&d_opacityConst, &this->opacityConstReal, sizeof(float));

  cudaMemcpyToSymbol(&d_tfComboSelectedColor, &this->tfComboSelectedColor, sizeof(int));

}

Widget::~Widget() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  free(this->fps);
}
