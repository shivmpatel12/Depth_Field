cmake_minimum_required(VERSION 3.6)
project(proj4_awn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(proj4_awn ${SOURCE_FILES})