cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(cudapractice LANGUAGES CXX CUDA)

add_executable(main main.cu)