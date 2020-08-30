cmake_minimum_required(VERSION 3.14)

add_hipacc_sample_cpu(HIDE) # no reference, long execution time
#add_hipacc_sample_cuda()
add_hipacc_sample_opencl(HIDE CPU GPU) # no reference
