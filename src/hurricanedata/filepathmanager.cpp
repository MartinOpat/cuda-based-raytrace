#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

#include "filepathmanager.h"

namespace fs = std::filesystem;

FilePathManager::FilePathManager(const std::string& path) {
    // Check if the directory exists
    if (!fs::exists(path) || !fs::is_directory(path)) {
        std::cerr << "Error: Directory does not exist or is not valid." << std::endl;
        exit(1);
    }

    // Iterate over directory entries
    for (const auto& entry : fs::directory_iterator(path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".nc4") {
            fileNames.push_back(entry.path().string());
        }
    }

    // Sort file names lexicographically
    std::sort(fileNames.begin(), fileNames.end());

    // Print sorted file names
    std::cout << "Files in directory '" << path << "' sorted lexicographically:\n";
    for (const auto& fileName : fileNames) {
        std::cout << fileName << std::endl;
    }
}

const char* FilePathManager::getPath(size_t index) const {
    return fileNames.at(index).c_str();
}


FilePathManager::~FilePathManager() { }