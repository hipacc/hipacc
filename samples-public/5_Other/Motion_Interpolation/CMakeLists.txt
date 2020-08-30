cmake_minimum_required(VERSION 3.14)

add_hipacc_sample_cpu()
add_hipacc_sample_cuda()
add_hipacc_sample_opencl(HIDE CPU) # compilation takes 15min, exclude from tests
add_hipacc_sample_opencl(GPU)
