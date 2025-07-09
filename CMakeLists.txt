cmake_minimum_required(VERSION 3.15)
project(Synapselib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add pybind11
add_subdirectory(external/pybind11)

# Create the Python extension module
add_library(Synapselib MODULE
        library.cpp
        bindings.cpp
)

# Include headers if needed
target_include_directories(Synapselib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Link pybind11
target_link_libraries(Synapselib PRIVATE pybind11::module)

# Remove 'lib' prefix
set_target_properties(Synapselib PROPERTIES PREFIX "")

# Set suffix based on platform
if (WIN32)
    set_target_properties(Synapselib PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(Synapselib PROPERTIES SUFFIX ".so")
endif()
