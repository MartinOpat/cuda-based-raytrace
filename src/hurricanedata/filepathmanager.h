#ifndef FILEPATHMANAGER_H
#define FILEPATHMANAGER_H

#include <string>
#include <vector>

class FilePathManager {
public:
    FilePathManager(const std::string& path);
    size_t getNumberOfFiles();
    const char* getPath(size_t index) const;

    ~FilePathManager();

private:
    std::vector<std::string> fileNames;
};

#endif //FILEPATHMANAGER_H