set(TRIQS_PATH /home/stephan/libs/triqs)
# Append triqs installed files to the cmake load path
list(APPEND CMAKE_MODULE_PATH ${TRIQS_PATH}/share/triqs/cmake)

# Start configuration
cmake_minimum_required(VERSION 2.8)
project(myproj CXX)
set(CMAKE_BUILD_TYPE Release)

# Load TRIQS, including all predefined variables from TRIQS installation
find_package(TRIQS REQUIRED)

# Linking and include info
link_libraries(${TRIQS_LIBRARY_ALL})
include_directories(${TRIQS_INCLUDE_ALL} /home/stephan/libs/eigen)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -std=c++11")

# Create executable
add_executable(ctint ctint.cpp main.cpp)
triqs_set_rpath_for_target(ctint)
