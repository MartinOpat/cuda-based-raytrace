#ifndef DATAREADER_H
#define DATAREADER_H

#include <string>
#include <experimental/propagate_const>
#include <memory>

#include "filepathmanager.h"


/**
 * @brief Simple wrapper around all the NetCDF functionality.
 */
class DataReader {
public:
    /**
     * @brief Constructor.
     * @param path Path to the directory containing all the .nc4 files.
     * @param variableName The variable we are interested in.
     */
    DataReader(const std::string &path, std::string variableName);

    /**
     * @brief The length of the flat data in variableName.
     * Used for allocating the right amount of memory.
     */
    size_t fileLength(size_t fileIndex);

    /**
     * @brief Write all the data in file fileIndex of variable we re interested in into the dataOut.
     */
    template <typename T>
    void loadFile(T* dataOut, size_t fileIndex);

    /**
     * @brief Write all the data in file fileIndex of variable variableName into the dataOut.
     * @param dataOut pointer to memory that should be written to.
     * @param fileIndex the index of the file we want to load into memory.
     * @param variableName the name of the variable
     */
    template <typename T>
    void loadFile(T* dataOut, size_t fileIndex, const std::string& variableName);

    /**
     * @brief Get size of a variable.
     */
    size_t axisLength(size_t fileIndex, const std::string& axisName);

    ~DataReader();
private:
    FilePathManager filePathManager;
    std::string variableName;
};


#endif //DATAREADER_H
