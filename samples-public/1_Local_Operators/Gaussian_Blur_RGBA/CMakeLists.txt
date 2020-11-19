cmake_minimum_required(VERSION 3.14)

add_hipacc_sample_dsl()
add_hipacc_sample_cpu() # vectorization disabled, due to unsupported vector type 'uchar4'
add_hipacc_sample_cuda(LOCAL PPT GRAPH)
add_hipacc_sample_opencl(CPU GPU)
