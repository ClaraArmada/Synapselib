cmake_minimum_required(VERSION 3.15)
project(Synapselib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add pybind11 as a subdirectory
add_subdirectory(external/pybind11)

# Create your Python extension module and link pybind11
add_library(Synapselib MODULE library.cpp)

target_link_libraries(Synapselib PRIVATE pybind11::module)

# Remove 'lib' prefix (Windows needs plain .pyd)
set_target_properties(Synapselib PROPERTIES PREFIX "")

if (WIN32)
    set_target_properties(Synapselib PROPERTIES SUFFIX ".pyd")
endif()

