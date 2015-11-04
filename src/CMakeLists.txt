# Start configuration
cmake_minimum_required(VERSION 2.8)
project(ctint)
MESSAGE("MCLL_SYSTEM_INFO set to $ENV{MCLL_SYSTEM_INFO}")
set(mode "mpi")

enable_language(CXX C)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Add load leveller source directory
set(MCLL "$ENV{HOME}/mc/load_leveller/trunk")
MESSAGE("load_leveller source directory set to ${MCLL}")

# Linking and include info
include_directories(${CMAKE_SOURCE_DIR})
include_directories($ENV{HOME}/libs/eigen $ENV{HOME}/eigen)
include_directories($ENV{HOME}/armadillo-6.200.2/include)
include_directories(${MCLL})

if(mode STREQUAL "mpi")
	# Set up MPI
	MESSAGE("Using MPI implementation.")
	FIND_PACKAGE(MPI REQUIRED)
	if (MPI_CXX_FOUND)
		set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
		set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS}")
		MESSAGE("Using mpi wrapper ${MPI_CXX_COMPILER}")
		
		include_directories(${MPI_CXX_INCLUDE_PATH})
	else()
		MESSAGE("MPI not found.")
	endif()
	
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

FIND_PACKAGE( Boost 1.40 )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
find_library(MPFR_LIBRARIES mpfr PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})
find_library(GMP_LIBRARIES gmp PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR})

set(BLAS_DIR /usr/lib/)
FIND_PACKAGE(BLAS)
set(LAPACK_DIR /usr/lib/)
FIND_PACKAGE(LAPACK)
find_library(ARPACK_LIBRARY NAMES arpack PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)
IF (ARPACK_LIBRARY)
	MESSAGE("ARPACK library found: ${ARPACK_LIBRARY}")
ELSE ()
	MESSAGE("ARPACK library not found.")
ENDIF ()
find_library(ARMADILLO_LIBRARY NAMES armadillo PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib $ENV{HOME}/armadillo-6.200.2 REQUIRED)
IF (ARMADILLO_LIBRARY)
	MESSAGE("Armadillo library found: ${ARMADILLO_LIBRARY}")
ELSE ()
	MESSAGE("Armadillo library not found.")
ENDIF ()

# Compiler flags
set(DEFINES "${DEFINES} -DMCL_DUMP_BUFFER=0 -DMCL_MEASUREMENTS_APPEND=0 -DMCL_MCL_RNG_MT")
if("$ENV{MCLL_SYSTEM_INFO}" STREQUAL "rwthcluster")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DMKL_DIRECT_CALL -ip -axCORE-AVX2,AVX,SSE4.2,SSE4.1 -fp-model fast=2 -std=c++11 ${DEFINES}")
elseif("$ENV{MCLL_SYSTEM_INFO}" STREQUAL "juqueen")
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -qarch=qp -qtune=qp -qstrict -qlanglvl=extended0x ${DEFINES}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -march=native -flto -fwhole-program -std=c++11 ${DEFINES}")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -march=native -flto -fwhole-program -std=c++11 ${DEFINES}")
endif()

# Create executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
add_executable(ctint ${LL_SRC} mc.cpp)
target_link_libraries(ctint ${MPI_LIBRARIES})
if(Boost_FOUND AND BLAS_FOUND AND LAPACK_FOUND AND MPFR_LIBRARIES AND GMP_LIBRARIES AND ARPACK_LIBRARY AND ARMADILLO_LIBRARY)
	MESSAGE("bbb")
	add_executable(ed ed.cpp)
	target_link_libraries(ed ${Boost_LIBRARIES} ${MPFR_LIBRARIES} ${GMP_LIB} ${LAPACK} ${BLAS} ${ARPACK_LIBRARY} ${ARMADILLO_LIBRARY})
endif()
