cmake_minimum_required(VERSION 3.14)

add_hipacc_sample_dsl()
if(WIN32)
    add_hipacc_sample_cpu() # vectorization disabled, due to unsupported vector type 'uchar4'
endif()
add_hipacc_sample_cuda(LOCAL PPT GRAPH)
add_hipacc_sample_opencl(CPU GPU)
