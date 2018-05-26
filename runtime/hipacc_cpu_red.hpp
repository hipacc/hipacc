#ifdef USE_OPENMP
#  include <omp.h>
#  define GET_NUM_CORES omp_get_num_procs()
#  define GET_THREAD_ID omp_get_thread_num()
#  define OPENMP_PRAGMA _Pragma("omp parallel for")
#else
#  define GET_NUM_CORES 1
#  define GET_THREAD_ID 0
#  define OPENMP_PRAGMA
#endif


#define REDUCTION_CPU_2D(NAME, DATA_TYPE, REDUCE, WIDTH, HEIGHT, PPT) \
inline DATA_TYPE NAME ##Kernel(DATA_TYPE input[HEIGHT][WIDTH], int width, int height, int stride, int offset_x=0, int offset_y=0) { \
    int num_cores = GET_NUM_CORES; \
 \
    DATA_TYPE* part_result = new DATA_TYPE[num_cores]; \
    int* init = new int[num_cores]; \
 \
    for (int tid = 0; tid < num_cores; ++tid) { \
        init[tid] = 1; \
    } \
 \
    int end = height/PPT; \
 \
    OPENMP_PRAGMA \
    for (int gid_y = 0; gid_y < end; ++gid_y) { \
        const int tid = GET_THREAD_ID; \
        int y = offset_y + gid_y * PPT; \
        if (init[tid] == 1) part_result[tid] = input[y][offset_x]; \
 \
        for (int p = 0; p < PPT; ++p) { \
            int gy = y + p; \
            for (int gid_x = offset_x + init[tid]; gid_x < offset_x + width; ++gid_x) { \
                part_result[tid] = REDUCE(part_result[tid], input[gy][gid_x]); \
            } \
            init[tid] = 0; \
        } \
    } \
 \
    if (int missing = height%PPT) { \
       int gid_y = offset_y + end * PPT; \
 \
       OPENMP_PRAGMA \
       for (int m = 0; m < missing; ++m) { \
            const int tid = GET_THREAD_ID; \
            int gy = gid_y + m; \
            for (int gid_x = offset_x; gid_x < offset_x + width; ++gid_x) { \
                part_result[tid] = REDUCE(part_result[tid], input[gy][gid_x]); \
            } \
       } \
    } \
 \
    for (int i = 1; i < num_cores; ++i) { \
        part_result[0] = REDUCE(part_result[0], part_result[i]); \
    } \
 \
    DATA_TYPE result = part_result[0]; \
 \
    delete [] init; \
    delete [] part_result; \
 \
    return result; \
}


#define BINNING_CPU_2D(NAME, DATA_TYPE, BIN_TYPE, REDUCE, BINNING, WIDTH, HEIGHT, PPT) \
inline void BINNING ##Put(BIN_TYPE *_lmem, uint _offset, uint idx, BIN_TYPE val) { \
    if (idx < _offset) { \
        _lmem[idx] = REDUCE(_lmem[idx], val); \
    } \
} \
 \
inline BIN_TYPE* NAME ##Kernel(DATA_TYPE input[HEIGHT][WIDTH], uint num_bins, int width, int height, int stride, int offset_x=0, int offset_y=0) { \
    int num_cores = GET_NUM_CORES; \
 \
    BIN_TYPE *bins = new BIN_TYPE[num_bins](); \
    BIN_TYPE *lbins = new BIN_TYPE[num_cores * num_bins](); \
    int end = height/PPT; \
 \
    OPENMP_PRAGMA \
    for (int gid_y = 0; gid_y < end; ++gid_y) { \
        const int tid = GET_THREAD_ID; \
        int y = offset_y + gid_y * PPT; \
 \
        for (int p = 0; p < PPT; ++p) { \
            int gy = y + p; \
            for (int gid_x = offset_x; gid_x < offset_x + width; ++gid_x) { \
                BINNING(&lbins[tid * num_bins], num_bins, num_bins, gid_x, gy, input[gy][gid_x]); \
            } \
        } \
    } \
 \
    if (int missing = height%PPT) { \
       int gid_y = offset_y + end * PPT; \
 \
       OPENMP_PRAGMA \
       for (int m = 0; m < missing; ++m) { \
            const int tid = GET_THREAD_ID; \
            int gy = gid_y + m; \
            for (int gid_x = offset_x; gid_x < offset_x + width; ++gid_x) { \
                BINNING(&lbins[tid * num_bins], num_bins, num_bins, gid_x, gy, input[gy][gid_x]); \
            } \
       } \
    } \
 \
    OPENMP_PRAGMA \
    for (uint i = 0; i < num_bins; ++i) { \
        for (int tid = 0; tid < num_cores; ++tid) { \
            bins[i] = REDUCE(bins[i], lbins[tid * num_bins + i]); \
        } \
    } \
 \
    delete [] lbins; \
 \
    return bins; \
}
