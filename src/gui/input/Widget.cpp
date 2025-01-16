#include "Widget.h"
#include "linalg/linalg.h"
#include "consts.h"
#include <cstdio>
#include <stdlib.h>
#include <ctime>

// probably not the cleanest way to do this but it works
void parseDate(char* string, int dayOfYear) {
  switch (dayOfYear) {
    case 1 ... 31:
      sprintf(string, "Jan %d", dayOfYear);
      break;
    case 32 ... 60:
      sprintf(string, "Feb %d", dayOfYear-31);
      break;
    case 61 ... 91:
      sprintf(string, "Mar %d", dayOfYear-60);
      break;
    case 92 ... 121:
      sprintf(string, "Apr %d", dayOfYear-91);
      break;
    case 122 ... 152:
      sprintf(string, "May %d", dayOfYear-121);
      break;
    case 153 ... 182:
      sprintf(string, "Jun %d", dayOfYear-152);
      break;
    case 183 ... 213:
      sprintf(string, "Jul %d", dayOfYear-182);
      break;
    case 214 ... 244:
      sprintf(string, "Aug %d", dayOfYear-213);
      break;
    case 245 ... 274:
      sprintf(string, "Sep %d", dayOfYear-244);
      break;
    case 275 ... 305:
      sprintf(string, "Oct %d", dayOfYear-274);
      break;
    case 306 ... 335:
      sprintf(string, "Nov %d", dayOfYear-305);
      break;
    case 336 ... 366:
      sprintf(string, "Dec %d", dayOfYear-335);
      break;
    default:
      sprintf(string, "∞");
  }
}

Widget::Widget(GLFWwindow* window) : 
  opacityK(0),
  sigmoidShift(0.5f),
  sigmoidExp(-250.0f),
  tfComboSelected(0),
  dateChanged(false),
  paused(true),
  renderOnce(false),
  bgColor(Color3::init(0.1f, 0.1f, 0.1f)),
  date(0),
  samplesPerPixel(1),
  alphaAcumLimit(0.4f),
  opacityConst(100),
  showSilhouettes(true),
  silhouettesThreshold(0.2f)
{
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  this->io = ImGui::GetIO();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init();

  this->fps = (char*)malloc(512*sizeof(char));
  this->dateString = (char*)malloc(512*sizeof(char));
  sprintf(this->dateString, "January 1st");

  resetCamera();
  resetLight();
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
  

  ImGui::Begin("Transfer Function Controls");
  ImGui::DragInt("Grad. exp. (log [1e-10, 1])", &this->opacityK, 1, 0, 100, "%d%%", ImGuiSliderFlags_AlwaysClamp);
  ImGui::DragFloat("Sig. shift", &this->sigmoidShift, 0.01f, 0.0f, 1.0f, "%.2f");
  ImGui::InputFloat("Sig. sxp", &this->sigmoidExp, 10.0f, 100.0f, "%.0f");
  ImGui::DragFloat("Alpha accum. limit", &this->alphaAcumLimit, 0.01f, 0.0f, 1.0f, "%.2f");
  ImGui::DragInt("Opacity const. (log [1e-5, 1])", &this->opacityConst, 1, 0, 100, "%d%%", ImGuiSliderFlags_AlwaysClamp);
  
  // the items[] contains the entries for the combobox. The selected index is stored as an int on this->tfComboSelected
  // the default entry is set in the constructor, so if you want that to be a specific entry just change it
  // whatever value is selected here is available on the gpu as d_tfComboSelected.
  const char* items[] = {"Opacity - gradient", "Opacity - sigmoid", "Opacity - constant", "..."};
  if (ImGui::BeginCombo("Transfer function", items[this->tfComboSelected])) {
    for (int n = 0; n < IM_ARRAYSIZE(items); n++) {
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


  float min = -1, max = 1;
  ImGui::Begin("Light Controls");
  ImGui::DragScalar("X coordinate", ImGuiDataType_Double, &this->lightPos.x, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Y coordinate", ImGuiDataType_Double, &this->lightPos.z, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Z coordinate", ImGuiDataType_Double, &this->lightPos.y, 0.5f, &min, &max, "%.3f");
  if (ImGui::Button("Reset light")) resetLight();
  ImGui::End();


  ImGui::Begin("Simulation Controls");
  if (ImGui::Button(this->paused ? "Unpause" : "Pause")) this->paused = !this->paused;
  ImGui::SameLine();
  if (ImGui::Button("Render once")) {
    this->paused = !this->paused;
    this->renderOnce = true;
  }
  sprintf(this->fps, "%.3f fps\n", fps);
  ImGui::Text(this->fps);

  // we have data from 2012-01-01 til 2012-09-11 == 255 days
  ImGui::SetNextItemWidth(20.0f * ImGui::GetFontSize()); 
  if (ImGui::SliderInt("Day of year", &this->date, 1, 255, "%d", ImGuiSliderFlags_NoInput)) {
    this->dateChanged = true;
    parseDate(this->dateString, this->date);
  }
  ImGui::SameLine();
  ImGui::Text(this->dateString);
  ImGui::DragInt("Samples per pixel", &this->samplesPerPixel, 1, 1, 16, "%d", ImGuiSliderFlags_AlwaysClamp);
  ImGui::End();


  ImGui::Begin("Camera Controls");
  ImGui::DragScalar("X position", ImGuiDataType_Double, &this->cameraPos.x, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Y position", ImGuiDataType_Double, &this->cameraPos.z, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("Z position", ImGuiDataType_Double, &this->cameraPos.y, 0.5f, &min, &max, "%.3f");
  ImGui::DragScalar("X direction", ImGuiDataType_Double, &this->cameraDir.x, 0.005f, &min, &max, "%.3f"); // TODO: should wrap around once fully rotated
  ImGui::DragScalar("Y direction", ImGuiDataType_Double, &this->cameraDir.z, 0.005f, &min, &max, "%.3f");
  ImGui::DragScalar("Z direction", ImGuiDataType_Double, &this->cameraDir.y, 0.005f, &min, &max, "%.3f");
  if (ImGui::Button("Reset camera")) resetCamera();
  ImGui::End();
  
  copyToDevice();
}

void Widget::render() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Widget::resetCamera() {
  this->cameraPos = Point3::init(50.0f, -50.0f, -75.0f);  // Camera for partially trimmed data set
  // this->cameraPos = Point3::init(300.0f, 200.0f, -700.0f);  // Camera for full data set
  
  Vec3 h_center = Vec3::init((float)VOLUME_WIDTH/2.0f, (float)VOLUME_HEIGHT/2.0f, (float)VOLUME_DEPTH/2.0f);
  this->cameraDir = (h_center - this->cameraPos).normalize();
}


void Widget::resetLight() {
  this->lightPos = Point3::init(1.5, 2.0, -1.0);
}

void Widget::copyToDevice() {
  cudaMemcpyToSymbol(&d_cameraPos, &this->cameraPos, sizeof(Point3));
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

  this->opacityConstReal = std::pow(10.0f, (-5 + 0.05 * this->opacityConst));
  cudaMemcpyToSymbol(&d_opacityConst, &this->opacityConstReal, sizeof(float));

  cudaMemcpyToSymbol(&d_tfComboSelectedColor, &this->tfComboSelectedColor, sizeof(int));
}

Widget::~Widget() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  free(this->fps);
  free(this->dateString);
}
