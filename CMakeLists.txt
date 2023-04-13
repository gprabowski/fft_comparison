cmake_minimum_required(VERSION 3.22)
project(fft_comparison LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE Release)

add_executable(fft_comparison utils/utils.cpp main.cpp)


target_include_directories(fft_comparison PUBLIC utils fft)

target_compile_options(fft_comparison PUBLIC "--std=c++20")
target_compile_options(fft_comparison PUBLIC "-mavx2")
target_compile_options(fft_comparison PUBLIC "-march=native")

# FFTW
include("cmake/CMakeFFTWConfig.txt")

find_package(FFTW)

target_include_directories(fft_comparison PUBLIC ${FFTW_INCLUDE_DIRS})
target_link_libraries(fft_comparison PUBLIC ${FFTW_LIBRARIES})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fft_comparison PUBLIC OpenMP::OpenMP_CXX)
endif()
