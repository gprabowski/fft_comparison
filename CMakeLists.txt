cmake_minimum_required(VERSION 3.22)
project(fft_comparison LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE Release)

set(FFTW_USE_STATIC_LIBS true)

add_executable(fft_comparison utils/utils.cpp 
    fft/fftw/fftw.cpp 
    fft/kissfft/kissfft.cpp 
    fft/ibb/ibb.cpp 
    fft/edp/edp.cpp 
    main.cpp)


target_include_directories(fft_comparison PUBLIC ${CMAKE_SOURCE_DIR} utils fft thirdparty/kissfft)

target_compile_options(fft_comparison PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:"-Xcompiler=-march=native">
                                              $<$<COMPILE_LANGUAGE:CUDA>:"--allow-unsupported-compiler">)


# FFTW
include("cmake/CMakeFFTWConfig.txt")

find_package(FFTW)

set(KISSFFT_DATATYPE "double")
add_subdirectory(thirdparty/kissfft)

target_include_directories(fft_comparison PUBLIC ${FFTW_INCLUDE_DIRS})
target_link_libraries(fft_comparison PUBLIC ${FFTW_LIBRARIES} kissfft)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fft_comparison PUBLIC OpenMP::OpenMP_CXX)
endif()
