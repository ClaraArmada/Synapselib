cmake_minimum_required(VERSION 3.31)
project(Synapselib)

set(CMAKE_CXX_STANDARD 20)

add_library(SynapselibLib STATIC library.cpp)
add_executable(Synapselib library.cpp)
