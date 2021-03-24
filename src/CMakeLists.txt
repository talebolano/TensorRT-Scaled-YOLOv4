project(Tnyolo)

include_directories(
    ./
    ../include
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
     )# 将给定目录添加到编译器

link_directories(${OpenCV_LIBRARIES_DIRS})
# set(CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
set(SRC_LIST resize.cu post.cu nms.cpp EntroyCalibrator.cpp trt.cpp )
add_library(Tnyolo SHARED ${SRC_LIST} )
target_link_libraries(Tnyolo ${TENSORRT_INFER} ${TENSORRT_ONNX} ${TENSORRT_PLUGIN} ${OpenCV_LIBS} Mish_plugin)