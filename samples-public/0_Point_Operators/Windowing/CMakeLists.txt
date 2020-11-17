cmake_minimum_required(VERSION 3.14)

get_filename_component(SAMPLE_NAME "${CMAKE_CURRENT_LIST_DIR}" NAME)

project(${SAMPLE_NAME})

add_hipacc_sample_dsl()
add_hipacc_sample_cpu(SSE4.2 AVX AVX2)
add_hipacc_sample_cuda(PPT GRAPH)
add_hipacc_sample_opencl(CPU GPU)
