cmake_minimum_required(VERSION 3.14)
project(quant_gru_cpu VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(LIB_SOURCES
    src/gru_forward_cpu_quant.cc
    src/quantize_lut.cc
)

# 静态库
add_library(quant_gru_cpu_static STATIC ${LIB_SOURCES})
target_include_directories(quant_gru_cpu_static PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 共享库
add_library(quant_gru_cpu_shared SHARED ${LIB_SOURCES})
target_include_directories(quant_gru_cpu_shared PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 示例程序
add_executable(gru_cpu_example example/main.cc)
target_link_libraries(gru_cpu_example PRIVATE quant_gru_cpu_static)

# 安装规则
install(TARGETS quant_gru_cpu_static quant_gru_cpu_shared
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

