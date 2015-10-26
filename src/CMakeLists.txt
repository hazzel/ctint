# Start configuration
cmake_minimum_required(VERSION 2.8)
project(ctint CXX)
set(CMAKE_BUILD_TYPE Release)
set(mode "mpi")

MESSAGE("MCLL_SYSTEM_INFO set to $ENV{MCLL_SYSTEM_INFO}")
# Add load leveller source directory
set(MCLL "$ENV{HOME}/mc/load_leveller/trunk")
MESSAGE("load_leveller source directory set to ${MCLL}")

if(mode STREQUAL "mpi")
	# Set up MPI
	MESSAGE("Using mpi implementation.")
	find_package(MPI REQUIRED)
	set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
	set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
	include_directories(${MPI_INCLUDE_PATH})
	
	file(GLOB LL_SRC ${MCLL}/dump.cpp 
						${MCLL}/parser.cpp
						${MCLL}/measurements.cpp
						${MCLL}/evalable.cpp
						${MCLL}/observable.cpp
						${MCLL}/random.cpp
						${MCLL}/main.cpp
						${MCLL}/runner.cpp)
elseif(mode STREQUAL "single")
	MESSAGE("Using single core implementation.")
	file(GLOB LL_SRC ${MCLL}/dump.cpp 
						${MCLL}/parser.cpp
						${MCLL}/measurements.cpp
						${MCLL}/evalable.cpp
						${MCLL}/observable.cpp
						${MCLL}/random.cpp
						${MCLL}/main.cpp
						${MCLL}/runner_single.cpp)
	set(DEFINES "${DEFINES} -DMCL_SINGLE")
endif()

# Linking and include info
include_directories(${CMAKE_SOURCE_DIR})
include_directories($ENV{HOME}/libs/eigen $ENV{HOME}/eigen)
include_directories(${MCLL})

# Compiler flags
set(DEFINES "${DEFINES} -DMCL_DUMP_BUFFER=0 -DMCL_MEASUREMENTS_APPEND -DMCL_MCL_RNG_MT")
if("$ENV{MCLL_SYSTEM_INFO}" STREQUAL "rwthcluster")
	set(CMAKE_CXX_FLAGS "-O3 -DMKL_DIRECT_CALL -ip -axCORE-AVX2,AVX,SSE4.2,SSE4.1 -fp-model fast=2 -std=c++11 ${DEFINES}")
else()
	set(CMAKE_CXX_FLAGS "-Ofast -ffast-math -march=native -flto -fwhole-program -std=c++11 ${DEFINES}")
endif()

# Create executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
add_executable(ctint ${LL_SRC} mc.cpp)
if(mode STREQUAL "mpi")
	target_link_libraries(ctint ${MPI_LIBRARIES})
endif()