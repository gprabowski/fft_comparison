cmake_minimum_required(VERSION 3.22)
project(fft_comparison LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)

add_executable(fft_comparison main.cpp)

target_include_directories(fft_comparison PUBLIC utils fft)

# FFTW
include("cmake/CMakeFFTWConfig.txt")

find_package(FFTW)

target_include_directories(fft_comparison PUBLIC ${FFTW_INCLUDE_DIRS})
target_link_libraries(fft_comparison PUBLIC ${FFTW_LIBRARIES})
