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
  this->sigmoidShift = 0.5f;
  this->sigmoidExp = -250.0f;
  this->tfComboSelected = 0;
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
  ImGui::DragInt("Gradient exp. (log [1e-10, 1])", &this->opacityK, 1, 0, 100, "%d%%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::DragFloat("sigmoidShift", &this->sigmoidShift, 0.01f, 0.0f, 1.0f, "%.2f");
  ImGui::InputFloat("sigmoidExp", &this->sigmoidExp, 10.0f, 100.0f, "%.0f");
  
  // the items[] contains the entries for the combobox. The selected index is stored as an int on this->tfComboSelected
  // the default entry is set in the constructor, so if you want that to be a specific entry just change it
  // whatever value is selected here is available on the gpu as d_tfComboSelected.
  const char* items[] = {"Opacity - gradient", "Opacity - sigmoid", "Opacity - constant", "..."};
  if (ImGui::BeginCombo("ComboBox for transferFunction", items[this->tfComboSelected]))
  {
    for (int n = 0; n < IM_ARRAYSIZE(items); n++)
    {
      const bool is_selected = (this->tfComboSelected == n);
      if (ImGui::Selectable(items[n], is_selected))
        this->tfComboSelected = n;
      if (is_selected)
        ImGui::SetItemDefaultFocus();
    }
    ImGui::EndCombo();
  }
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

  cudaMemcpyToSymbol(&d_sigmoidShift, &this->sigmoidShift, sizeof(float));
  cudaMemcpyToSymbol(&d_sigmoidExp, &this->sigmoidExp, sizeof(float));
  cudaMemcpyToSymbol(&d_tfComboSelected, &this->tfComboSelected, sizeof(float));
}

Widget::~Widget() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  free(this->fps);
}
