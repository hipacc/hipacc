cmake_minimum_required(VERSION 3.14)

add_hipacc_sample_dsl()
add_hipacc_sample_cpu(SSE4.2 AVX AVX2)
add_hipacc_sample_cuda(LOCAL PPT GRAPH)
add_hipacc_sample_opencl(CPU GPU)
