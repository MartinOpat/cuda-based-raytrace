#include "MainWindow.h"

#include "cuda_runtime.h"
#include <iostream>
#include <memory>

#include "Shader.h"

Window::Window(unsigned int w, unsigned int h) {
  this->w = w;
  this->h = h;
}

void framebuffer_size_callback(GLFWwindow* window, int w, int h) {
  // This function is called by glfw when the window is reized.
  glViewport(0 , 0, w, h);
  Window* newWin = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
  newWin->resize(w, h);

}

int Window::init(float* data) {
  // init glfw
  glfwInit();
  // requesting context version 1.0 makes glfw try to provide the latest version if possible
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 1);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  this->window = glfwCreateWindow(this->w, this->h, "CUDA ray tracing", NULL, NULL);

  //hide cursor // TODO: switch from this style input to something more resembling an actual gui
  glfwSetInputMode(this->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetWindowUserPointer(this->window, reinterpret_cast<void*>(this));

  if (this->window == NULL) {
    std::cout << "Failed to create window\n";
    glfwTerminate();
    return -1;
  }

	glfwMakeContextCurrent(this->window);

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

  while (!glfwWindowShouldClose(window)) {
    Window::tick();
  }

  Window::free(data);
  return 0;
}


int Window::init_quad(float* data) {
  this->current_quad = std::make_unique<Quad>(this->w, this->h);
  this->current_quad->cuda_init(data);
  this->current_quad->make_fbo();

  this->shader = std::make_unique<Shader>("./shaders/vertshader.glsl", "./shaders/fragshader.glsl");
  this->shader->use();

  return 0;
}


void Window::free(float* data) {
  // To preserve the proper destruction order we forcefully set the quads to null (calling their destructor in the process)
  // Not strictly necessary, but i saw some weird errors on exit without this so best to keep it in.
  this->current_quad = nullptr;
  cudaFree(data);

  glfwDestroyWindow(window);
  glfwTerminate();
}


void Window::tick() {
  // manually track time diff
	std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  float diff = (float) std::chrono::duration_cast<std::chrono::milliseconds>(now - this->last_frame).count();
  this->last_frame = now;

  // TODO: remove debug line at some point
  std::cout << 1000.0/diff << " fps\n";

  // TODO: code input logic and class/struct and stuff
  // ticking input probably involves 4? steps:
  // * check if window needs to be closed (escape/q pressed)
  // * check if camera moved (wasd/hjkl pressed)
  // (phase 3/do later): check if we switched from realtime tracing to that other option - maybe a pause function? (p pressed?)
  // * if moved -> update camera (raytracing will involve some logic here too? see when i get there)
  
  // tick render
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

  // render frame
	glBindFramebuffer(GL_FRAMEBUFFER, this->current_quad->fb);
	this->current_quad->render();
  this->shader->use();
	glBindVertexArray(this->current_quad->VAO);
	glBindTexture(GL_TEXTURE_2D, this->current_quad->tex);
	glDrawArrays(GL_TRIANGLES, 0, 6); // draw current frame to texture
  
  // check for events
  // + swap buffers; TODO: check if necessary?
	glfwSwapBuffers(this->window);
	glfwPollEvents();

  std::cout << "done ticking\n";
  
}

void Window::resize(unsigned int w, unsigned int h) {
  this->w = w;
  this->h = h;
  this->current_quad->resize(w, h);
}
