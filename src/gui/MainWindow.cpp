#include "MainWindow.h"

#include "hurricanedata/datareader.h"
#include "cuda_runtime.h"
#include <csignal>
#include <iostream>
#include <memory>

#include <vector>
#include "consts.h"
#include "Shader.h"
#include "input/Widget.h"
#include "cuda_error.h"

// FIXME: this is the worst code in this project - very ad hoc
// this is a blocking operation, and really does not follow any practices of code design
// should really be a proper class like GpuBufferHandler.
void loadData(float* d_data, const int idx) {
  std::cout << "hi\n";

  std::vector<float> h_data;
  std::cout << "hi\n";
  std::string path = "data/trimmed";
  std::cout << "hi\n";
  std::string variable = "T";
  std::cout << "hi\n";

  DataReader dataReader(path, variable);
  std::cout << "hi\n";

  size_t dataLength = dataReader.fileLength(idx);
  std::cout << "hi\n";

  h_data.resize(dataLength);
  std::cout << "hi\n";

  dataReader.loadFile(h_data.data(), idx);
  std::cout << "hi\n";

  // getTemperature(h_data, idx); 
  // getSpeed(h_data, idx); 

  float* hostVolume = new float[VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH];
  for (int i = 0; i < VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH; i++) {
    hostVolume[i] = h_data[i + 0*VOLUME_DEPTH*VOLUME_HEIGHT*VOLUME_WIDTH];
    // Discard missing values
    if (h_data[i + 0*VOLUME_DEPTH*VOLUME_HEIGHT*VOLUME_WIDTH] + epsilon >= infty) hostVolume[i] = -infty;
  }
  std::cout << "hi\n";

  // Reverse the order of hostVolume - why is it upside down anyway?
  for (int i = 0; i < VOLUME_WIDTH; i++) {
    for (int j = 0; j < VOLUME_HEIGHT; j++) {
      for (int k = 0; k < VOLUME_DEPTH/2; k++) {
        float temp = hostVolume[i + j*VOLUME_WIDTH + k*VOLUME_WIDTH*VOLUME_HEIGHT];
        hostVolume[i + j*VOLUME_WIDTH + k*VOLUME_WIDTH*VOLUME_HEIGHT] = hostVolume[i + j*VOLUME_WIDTH + (VOLUME_DEPTH - 1 - k)*VOLUME_WIDTH*VOLUME_HEIGHT];
        hostVolume[i + j*VOLUME_WIDTH + (VOLUME_DEPTH - 1 - k)*VOLUME_WIDTH*VOLUME_HEIGHT] = temp;
      }
    }
  }
  std::cout << "hi\n";

  // Allocate + copy data to GPU
  size_t volumeSize = sizeof(float) * VOLUME_WIDTH * VOLUME_HEIGHT * VOLUME_DEPTH;
  std::cout << "hi\n";
  cudaMemcpy(d_data, hostVolume, volumeSize, cudaMemcpyHostToDevice);
  std::cout << "hi\n";
}

void Window::saveImage() {
  unsigned char* pixels = new unsigned char[this->w * this->h * 3];
  glReadPixels(0, 0, this->w, this->h, GL_RGB, GL_UNSIGNED_BYTE, pixels);
  const char* filename = "output.ppm"; // TODO: make this the current time

  std::ofstream imageFile(filename, std::ios::out | std::ios::binary);
  imageFile << "P6\n" << this->w << " " << this->h << "\n255\n";
  for (int i = 0; i < this->w * this->h * 3; i++) {
      imageFile << pixels[i];
  }
  imageFile.close();

  delete[] pixels;
}

Window::Window(unsigned int w, unsigned int h) {
  this->w = w;
  this->h = h;

  this->gpuPerf.open("gpuPerformance");
  this->cpuPerf.open("cpuPerformance");
}

void framebuffer_size_callback(GLFWwindow* window, int w, int h) {
  // This function is called by glfw when the window is resized.
  glViewport(0 , 0, w, h);
  Window* newWin = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  newWin->resize(w, h);
}

int Window::init(float* data) {
  this->data = data;

  // init glfw
  glfwInit();
  // requesting context version 1.0 makes glfw try to provide the latest version if possible
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 1);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  this->window = glfwCreateWindow(this->w, this->h, "CUDA ray tracing", NULL, NULL);


  if (this->window == NULL) {
    std::cout << "Failed to create window\n";
    glfwTerminate();
    return -1;
  }

	glfwMakeContextCurrent(this->window);
	glfwSetWindowUserPointer(this->window, reinterpret_cast<void*>(this));

  // init glad(opengl)
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD\n";
    return -1;
  }

  // init framebuffer
  glViewport(0, 0, this->w, this->h);
  if (glfwSetFramebufferSizeCallback(this->window, framebuffer_size_callback) != 0) return -1;

  if (init_quad(data)) return -1;
	this->last_frame = std::chrono::steady_clock::now();

  // init imGUI
  this->widget = new Widget(this->window);

  // loop function for draw calls etc.
  while (!glfwWindowShouldClose(window)) {
    Window::tick();
  }

  Window::free(data);
  return 0;
}


int Window::init_quad(float* data) {
  this->quad = std::make_unique<Quad>(this->w, this->h);
  this->quad->cuda_init(data);
  this->quad->make_fbo();

  this->shader = std::make_unique<Shader>("./shaders/vertshader.glsl", "./shaders/fragshader.glsl");
  this->shader->use();

	glUniform1i(glGetUniformLocation(this->shader->ID, "currentFrameTex"), 0);
  return 0;
}


void Window::free(float* data) {
  // To preserve the proper destruction order we forcefully set pointers to null (calling their destructor in the process)
  // Not strictly necessary, but i saw some weird errors on exit without this so best to keep it in.
  this->quad = nullptr;
  this->widget = nullptr;
  cudaFree(data);

  glfwDestroyWindow(window);
  glfwTerminate();

  this->gpuPerf.close();
  this->cpuPerf.close();
}


void Window::tick() {
  // manually track time diff
	std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  float diff = (float) std::chrono::duration_cast<std::chrono::milliseconds>(now - this->last_frame).count();
  this->last_frame = now;
  this->cpuPerf << diff << "\n";

  // input
  this->widget->tick(1000.0/diff);
  if (this->widget->dateChanged) {
    // TODO: Load new date file here
    loadData(this->data, this->widget->date);
    this->widget->dateChanged = false;
  }
  if (this->widget->saveImage) {
    saveImage();
    this->widget->saveImage = false;
  }
  
  // tick render
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

  if (!this->widget->paused){
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
    this->quad->render();

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float t;
  cudaEventElapsedTime(&t, start, stop);
  this->gpuPerf << t << "\n";

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  } 

  this->shader->use();
	glBindVertexArray(this->quad->VAO);
	glBindTexture(GL_TEXTURE_2D, this->quad->tex);
	glDrawArrays(GL_TRIANGLES, 0, 6); // draw current frame to texture
  
  // render ImGui context
  this->widget->render();

  // check for events
	glfwSwapBuffers(this->window);
	glfwPollEvents();

}

void Window::resize(unsigned int w, unsigned int h) {
  this->w = w;
  this->h = h;
  this->quad->resize(w, h);
}
