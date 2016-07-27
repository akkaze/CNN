cmake_minimum_required(VERSION 2.8)
set(project test)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=gnu++0x)
endif()

set(CMAKE_BUILD_TYPE Debug)  
SET(BOOST_DIR "/usr/include")
FILE(GLOB SRC_LIST "*.cpp")
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenBLAS REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
find_package(Boost COMPONENTS chrono thread date_time program_options filesystem system REQUIRED)

if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wno-unused-variable -pthread")
else( CMAKE_BUILD_TYPE STREQUAL "Debug" )
        set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -pthread -fopenmp")
endif( CMAKE_BUILD_TYPE STREQUAL "Debug" )

message("Build Type:"${CMAKE_BUILD_TYPE} ${CMAKE_CXX_FLAGS})
SET(EIGEN_UNSUPPORTED_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}/unsupported")
ADD_EXECUTABLE(${project} ${SRC_LIST})
include_directories(${BOOST_LIBRARYDIR} ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_PATH})

target_link_libraries(${project} ${OpenBLAS_LIB} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
