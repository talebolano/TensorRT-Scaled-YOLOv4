cmake_minimum_required(VERSION 2.8)

project(yolov4_cuda_test LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Relaese)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11 -Wall -Ofast")

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
################################CUDA GPU SET############################################################
if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      53
      61
      70
      75
    )
    string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
    if (IS_ARM)
    # Xavier (SM72) only supported for aarch64.
    list(APPEND GPU_ARCHS 72)
  endif()
  if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
    # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
    list(APPEND GPU_ARCHS 80)
  else()
    message(WARNING "Detected CUDA version is < 11.0. SM80 not supported.")
  endif()

  message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
endif()
foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
# Generate PTX for the last architecture in the list.
list(GET GPU_ARCHS -1 LATEST_SM)
set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
-D_FORCE_INLINES -Xcompiler -fPIC \
${GENCODES} \
")
##################################TENSORRT SET##########################################################
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES include) # 找目录
MESSAGE(STATUS "found tensorrt header at ${TENSORRT_INCLUDE_DIR}")
MESSAGE(STATUS "found cuda header so at ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

find_library(TENSORRT_INFER libnvinfer.so
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64) 

find_library(TENSORRT_ONNX libnvonnxparser.so
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN libnvinfer_plugin.so
            HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
            PATH_SUFFIXES lib lib64 lib/x64)

MESSAGE(STATUS "found tensorrt infer so at ${TENSORRT_INFER}")
MESSAGE(STATUS "found tensorrt onnx so at ${TENSORRT_ONNX}")
MESSAGE(STATUS "found tensorrt plugin so at ${TENSORRT_PLUGIN}")

add_subdirectory(mishplugin) 
add_subdirectory(src) 
add_subdirectory(mot) 

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/mot)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${OPENCV_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARIES_DIRS})

ADD_EXECUTABLE(makeCudaEngine cudaEngine.cpp )
target_link_libraries(makeCudaEngine ${OpenCV_LIBS} Tnyolo)

ADD_EXECUTABLE(inferYoloCuda inference.cpp )
target_link_libraries(inferYoloCuda ${OpenCV_LIBS} Tnyolo)

ADD_EXECUTABLE(inferYoloMot inference_mot.cpp )
target_link_libraries(inferYoloMot ${OpenCV_LIBS} Tnyolo SortMot)
