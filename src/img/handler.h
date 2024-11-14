#pragma once

#include <fstream>


void saveImage(const char* filename, unsigned char* framebuffer, int width, int height) {  // TODO: Figure out a better way to do this
    std::ofstream imageFile(filename, std::ios::out | std::ios::binary);
    imageFile << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; i++) {
        imageFile << framebuffer[i];
    }
    imageFile.close();
}