cmake_minimum_required(VERSION 3.14)

add_hipacc_sample_cpu(HIDE SSE4.2 AVX AVX2) # no reference
add_hipacc_sample_cuda(HIDE) # no reference
add_hipacc_sample_opencl(HIDE CPU GPU) # no reference
