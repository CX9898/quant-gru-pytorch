# Specify the minimum version of cmake
cmake_minimum_required(VERSION 2.8)

# Project Information
project(gru)

# Add languages needed when building the project
enable_language(CXX)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(Python3_ROOT_DIR "$ENV{HOME}/miniconda3/envs/quant-gru-pytouch")
set(Python3_EXECUTABLE "$ENV{HOME}/miniconda3/envs/quant-gru-pytouch/bin/python")
set(Torch_DIR "$ENV{HOME}/miniconda3/envs/quant-gru-pytouch/lib/python3.9/site-packages/torch/share/cmake/Torch")
set(pybind11_DIR "$ENV{HOME}/miniconda3/envs/quant-gru-pytouch/share/cmake/pybind11")
find_package(Python3 REQUIRED COMPONENTS Interpreter Development Development.Embed)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
message(STATUS "Python3 Include Dir: ${Python3_INCLUDE_DIRS}")  # 打印 Python 头文件路径
message(STATUS "Python3 Libraries: ${Python3_LIBRARIES}")        # 打印 Python 库路径

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Sets the C++ standard used by the project
# 设置 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)   # GNU 扩展，和 PyTorch 推荐一致

# CUDA 也使用 C++17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specifies which GPU architectures are supported when compiling CUDA code (optional)
#set(CMAKE_CUDA_ARCHITECTURES 86)

# Save the folder path in a variable
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# All source files in src folder are stored in SRC_FILES variable
file(GLOB SRC_FILES "${SRC_DIR}/*.c" "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.cc" "${SRC_DIR}/*.cxx" "${SRC_DIR}/*.cu")

# Output file list information
message(STATUS "Src files: ${SRC_FILES}")

# Add generate target
add_executable(${PROJECT_NAME})

#pybind11_add_module(gru_calibration_module MODULE bindings.cpp gru_calibration.cpp)

## Print kernel function register information
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Set the CUDA separable compilation property
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Link the source file to the build target
target_sources(${PROJECT_NAME} PRIVATE ${SRC_FILES})

# Set the installation Path
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")

# Set installation rules
install(TARGETS ${PROJECT_NAME} DESTINATION bin)

# Add header directory (locally)
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

# Linked cuda Runtime library
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# Linked cuBLAS library
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cublas)

# Linked cuFFT library
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cufft)

# Linked cuRAND library
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::curand)

# Linked cuSOLVER library
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cusolver)

# Linked cuSPARSE library
#target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cusparse)

target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${Python3_LIBRARIES})


target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

# Linked OpenMP library
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

# Eigen库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${TORCH_INCLUDE_DIRS}           # Torch 头文件（必须包含！）
        ${Python3_INCLUDE_DIRS}
)

message(STATUS "TORCH_INCLUDE_DIRS : ${TORCH_INCLUDE_DIRS}")
