cmake_minimum_required(VERSION 2.8)

project(yolov4_cuda_test LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Relaese)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Ofast")

add_subdirectory(src) 


find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURVE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${OPENCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARIES_DIRS})

ADD_EXECUTABLE(makeCudaEngine cudaEngine.cpp )
target_link_libraries(makeCudaEngine ${OpenCV_LIBS} Tnyolo )

ADD_EXECUTABLE(inferYoloCuda inference.cpp )
target_link_libraries(inferYoloCuda ${OpenCV_LIBS} Tnyolo )

