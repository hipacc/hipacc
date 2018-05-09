
#define REDUCTION_CPU_2D_OPENMP(NAME, DATA_TYPE, REDUCE, WIDTH, HEIGHT, PPT, CLEAR) \
inline DATA_TYPE NAME ##Kernel(DATA_TYPE input[HEIGHT][WIDTH], int width, int height, int stride) { \
    int num_cores = omp_get_num_procs(); \
 \
    DATA_TYPE result = CLEAR; \
    DATA_TYPE* part_result = new DATA_TYPE[num_cores](); \
 \
    int end = height/PPT; \
 \
    _Pragma("omp parallel for") \
    for (int gid_y = 0; gid_y < end; ++gid_y) { \
        const int tid = omp_get_thread_num(); \
        int y = gid_y * PPT; \
 \
        for (int p = 0; p < PPT; ++p) { \
            int gy = y + p; \
            for (int gid_x = 0; gid_x < width; ++gid_x) { /* vectorize */ \
                part_result[tid] = REDUCE(part_result[tid], input[gy][gid_x]); \
            } \
        } \
    } \
 \
    if (int missing = height%PPT) { \
       int gid_y = end*PPT; \
       for (int m = 0; m < missing; ++m) { \
            int gy = gid_y + m; \
            for (int gid_x = 0; gid_x < width; ++gid_x) { /* vectorize */ \
                part_result[0] = REDUCE(part_result[0], input[gy][gid_x]); \
            } \
       } \
    } \
 \
    for (int i = 1; i < num_cores; ++i) { \
        part_result[0] = REDUCE(part_result[0], part_result[i]); \
    } \
 \
    result = part_result[0]; \
 \
    delete [] part_result; \
 \
    return result; \
}

#define REDUCTION_CPU_2D_SINGLE(NAME, DATA_TYPE, REDUCE, WIDTH, HEIGHT, PPT, CLEAR) \
inline DATA_TYPE NAME ##Kernel(DATA_TYPE input[HEIGHT][WIDTH], int width, int height, int stride) { \
    DATA_TYPE result = CLEAR; \
 \
    int end = height/PPT; \
 \
    for (int gid_y = 0; gid_y < end; ++gid_y) { \
        int y = gid_y * PPT; \
 \
        for (int p = 0; p < PPT; ++p) { \
            int gy = y + p; \
            for (int gid_x = 0; gid_x < width; ++gid_x) { /* vectorize */ \
                result = REDUCE(result, input[gy][gid_x]); \
            } \
        } \
    } \
 \
    if (int missing = height%PPT) { \
       int gid_y = end*PPT; \
       for (int m = 0; m < missing; ++m) { \
            int gy = gid_y + m; \
            for (int gid_x = 0; gid_x < width; ++gid_x) { /* vectorize */ \
                result = REDUCE(result, input[gy][gid_x]); \
            } \
       } \
    } \
 \
    return result; \
}

