cmake_minimum_required(VERSION 3.28)
project(eva_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Keep -I flags inline; clangd cannot read nvcc --options-file response files.
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Learning repo: default to Debug when no build type is given.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
# Debug only adds host -g for nvcc; -G adds device debug info for cuda-gdb (disables device optimizations).
add_compile_options($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)

# Every .cu file under freeCodeCamp.org/ and leetgpu/ is a standalone program.
file(GLOB_RECURSE sources CONFIGURE_DEPENDS freeCodeCamp.org/*.cu leetgpu/*.cu)

foreach(src IN LISTS sources)
  # Target name from the relative path, e.g. "leetgpu/softmax/softmax.cu" -> leetgpu_softmax_softmax
  file(RELATIVE_PATH rel "${PROJECT_SOURCE_DIR}" "${src}")
  string(REGEX REPLACE "\\.cu$" "" name "${rel}")
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" name "${name}")
  add_executable(${name} "${src}")
  target_include_directories(${name} PRIVATE include)
endforeach()
