cmake_minimum_required(VERSION 3.14)
project(quant_gru_cpu VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

include_directories(${CMAKE_SOURCE_DIR}/include)

set(LIB_SOURCES
    src/gru_forward_cpu_quant.cc
    src/quantize_lut.cc
)

add_library(quant_gru_cpu_static STATIC ${LIB_SOURCES})
add_library(quant_gru_cpu_shared SHARED ${LIB_SOURCES})

add_executable(gru_cpu_example example/main.cc)
target_link_libraries(gru_cpu_example PRIVATE quant_gru_cpu_static)

install(TARGETS quant_gru_cpu_static quant_gru_cpu_shared
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
