cmake_minimum_required(VERSION 3.18)

project(quant-gru LANGUAGES CXX CUDA)

# ------------------------------
# C++ / CUDA 标准
# ------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)        # GNU 扩展
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ------------------------------
# CUDA 架构设置
# RTX 20系=75, RTX 30系=86, RTX 40系=89, RTX 50系=120
# ------------------------------
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)

# ------------------------------
# include / src 路径
# ------------------------------
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

# 所有源码文件
file(GLOB SRC_FILES
        "${SRC_DIR}/*.c"
        "${SRC_DIR}/*.cpp"
        "${SRC_DIR}/*.cc"
        "${SRC_DIR}/*.cxx"
        "${SRC_DIR}/*.cu"
)
message(STATUS "Src files: ${SRC_FILES}")

# ------------------------------
# CUDA / OpenMP
# ------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# ------------------------------
# 编译静态库和动态库
# ------------------------------
#add_library(gru_quant_static STATIC ${SRC_FILES})
add_library(gru_quant_shared SHARED ${SRC_FILES})

# 输出目录
#set_target_properties(gru_quant_static PROPERTIES
#        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pytorch/lib
#)
set_target_properties(gru_quant_shared PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pytorch/lib
        POSITION_INDEPENDENT_CODE ON
)

# ------------------------------
# [可选] CUDA 分离编译
# 用途：允许跨 .cu 文件调用 __device__ 函数，或链接 CUDA 库的 device API
# 代价：增加一个 device link 步骤，编译稍慢
# 当前项目：所有 __device__ 都是 inline，不需要开启
# ------------------------------
# set_target_properties(gru_quant_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ------------------------------
# Windows 特定配置
# ------------------------------
if(WIN32)
    set_target_properties(gru_quant_shared PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pytorch/lib  # Windows DLL 输出目录
            WINDOWS_EXPORT_ALL_SYMBOLS ON  # 自动导出所有符号，生成 .lib 导入库
    )
    # MSVC 使用符合标准的预处理器，消除 CUDA/CUB 警告
    if(MSVC)
        add_compile_options(/Zc:preprocessor)
    endif()
endif()

# ------------------------------
# 可执行程序
# ------------------------------
add_executable(gru_example "${CMAKE_SOURCE_DIR}/example/gru.cc")

# 让可执行程序链接共享库
target_link_libraries(gru_example PRIVATE gru_quant_shared)

# ------------------------------
# 包含目录
# ------------------------------
#target_include_directories(gru_quant_static PUBLIC ${INCLUDE_DIR})
target_include_directories(gru_quant_shared PUBLIC ${INCLUDE_DIR})
target_include_directories(gru_example PRIVATE ${INCLUDE_DIR})

# Torch / Python3 include (如果你需要调用 Torch API)
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# target_include_directories(gru_quant_shared PRIVATE ${Python3_INCLUDE_DIRS})

# ------------------------------
# eigen库
# ------------------------------
#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
#target_link_libraries(gru_example PRIVATE Eigen3::Eigen)

# ------------------------------
# 链接库
# ------------------------------
#target_link_libraries(gru_quant_static PUBLIC CUDA::cublas CUDA::cudart OpenMP::OpenMP_CXX)
target_link_libraries(gru_quant_shared PUBLIC CUDA::cublas CUDA::cudart OpenMP::OpenMP_CXX)
target_link_libraries(gru_example PRIVATE CUDA::cublas CUDA::cudart OpenMP::OpenMP_CXX)
