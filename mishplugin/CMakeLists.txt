project(Mish_plugin)

include_directories(
    ./
    ../include
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)# 将给定目录添加到编译器

set(SRC_LIST mishkernel.cu mish.cpp)
add_library(Mish_plugin SHARED ${SRC_LIST})
target_link_libraries(Mish_plugin ${TENSORRT_INFER} ${TENSORRT_PLUGIN} ${CUDA_LIBRARY})
