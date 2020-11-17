cmake_minimum_required(VERSION 3.14)

get_filename_component(SAMPLE_NAME "${CMAKE_CURRENT_LIST_DIR}" NAME)

project(${SAMPLE_NAME})

add_hipacc_sample_dsl()
add_hipacc_sample_cpu() # vectorization disabled, due to unsupported vector type 'uchar4'
add_hipacc_sample_cuda(PPT GRAPH)
add_hipacc_sample_opencl(CPU GPU)
