cmake_minimum_required(VERSION 3.24)
project(fft_comparison LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(fft_comparison main.cpp)

target_include_directories(fft_comparison PUBLIC utils fft)
