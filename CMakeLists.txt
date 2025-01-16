cmake_minimum_required (VERSION 3.8)

project ("cuda-raytracer" LANGUAGES CUDA CXX C)

# get the shader files in the build directory
file(COPY ${CMAKE_SOURCE_DIR}/src/gui/shaders DESTINATION .)

set(CMAKE_BUILD_TYPE Debug)

# source files
file(GLOB_RECURSE SOURCE_FILES 
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cpp
	${CMAKE_SOURCE_DIR}/src/*.cu)
	
# header files
file(GLOB HEADER_FILES 
	${CMAKE_SOURCE_DIR}/src/*.h
	${CMAKE_SOURCE_DIR}/src/*.hpp)

# imgui - note: could be done as a library as well but this was easier for now
file(GLOB IMGUI_FILES
  ${CMAKE_SOURCE_DIR}/include/imgui/*.h
  ${CMAKE_SOURCE_DIR}/include/imgui/*.cpp
  ${CMAKE_SOURCE_DIR}/include/imgui/backends/imgui_impl_glfw.*
  ${CMAKE_SOURCE_DIR}/include/imgui/backends/imgui_impl_opengl3.*)
include_directories(${CMAKE_SOURCE_DIR}/include/imgui)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${IMGUI_FILES})

# CUDA has specific architectures - set it to the system's architecture if available (or 70 by default)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 70)
set_target_properties(${PROJECT_NAME}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# Add .lib files
link_directories(${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")


# Package management

# OpenGL
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)

# GLFW
find_package(GLFW3 REQUIRED)
message(STATUS "Found GLFW3 in ${GLFW3_INCLUDE_DIR}")

# GLAD
add_library(GLAD "thirdparty/glad.c")

# CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# netcdf
find_package(netCDF REQUIRED)
message(STATUS "Found netcdf in ${NETCDF_INCLUDE_DIR}")

execute_process(
    COMMAND nc-config --includedir
    OUTPUT_VARIABLE NETCDF_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ncxx4-config --libdir
    OUTPUT_VARIABLE NETCDFCXX_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

target_include_directories(cuda-raytracer PUBLIC ${netCDF_INCLUDE_DIR})

find_library(NETCDF_LIB NAMES netcdf-cxx4 netcdf_c++4 PATHS ${NETCDFCXX_LIB_DIR} NO_DEFAULT_PATH)

set(LIBS ${GLFW3_LIBRARY} ${OPENGL_LIBRARY} GLAD ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${NETCDF_LIB})

include_directories(
	"${CMAKE_SOURCE_DIR}/src"
	"${CMAKE_SOURCE_DIR}/include"
	"${CMAKE_SOURCE_DIR}/include/glad"
)

target_link_libraries(${PROJECT_NAME} ${LIBS})


function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
	get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
	if(NOT "${old_flags}" STREQUAL "")
		string(REPLACE ";" "," CUDA_flags "${old_flags}")
		set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
			"$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
			)
	endif()
endfunction()

CUDA_CONVERT_FLAGS(${PROJECT_NAME})

# check for cache variable set in cmakepresets called IS_CUDA_DEBUG
if(IS_CUDA_DEBUG) 
	target_compile_options(${PROJECT_NAME} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")
	message(STATUS "CUDA_DEBUG is ON")
endif()

