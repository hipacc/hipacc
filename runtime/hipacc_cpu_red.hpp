#ifdef USE_OPENMP
# include <omp.h>
# define GET_NUM_CORES omp_get_num_procs()
# define GET_THREAD_ID omp_get_thread_num()
# ifdef _MSC_VER
#   define OPENMP_PRAGMA __pragma(omp parallel for)
#   define OPENMP_PRAGMA_TID __pragma(omp parallel for firstprivate(tid))
# else
#   define OPENMP_PRAGMA _Pragma("omp parallel for")
#   define OPENMP_PRAGMA_TID _Pragma("omp parallel for firstprivate(tid)")
# endif
#else
# define GET_NUM_CORES 1
# define GET_THREAD_ID 0
# define OPENMP_PRAGMA
# define OPENMP_PRAGMA_TID
#endif

#define REDUCTION_CPU_2D(NAME, DATA_TYPE, REDUCE, PPT)                         \
  inline DATA_TYPE NAME##Kernel(DATA_TYPE *input, int width,                   \
                                int height, int stride, int offset_x = 0,      \
                                int offset_y = 0) {                            \
    int tid = -1;                                                              \
    const int num_cores = GET_NUM_CORES;                                       \
                                                                               \
    std::vector<DATA_TYPE> part_result(num_cores);                             \
                                                                               \
    int end = height / PPT;                                                    \
                                                                               \
    OPENMP_PRAGMA_TID                                                          \
    for (int gid_y = 0; gid_y < end; ++gid_y) {                                \
      int y = offset_y + gid_y * PPT;                                          \
      int skip = 0;                                                            \
      if (tid == -1) {                                                         \
        tid = GET_THREAD_ID;                                                   \
        part_result[tid] = input[y * stride + offset_x];                       \
        skip = 1;                                                              \
      }                                                                        \
                                                                               \
      for (int p = 0; p < PPT; ++p) {                                          \
        int gy = y + p;                                                        \
        for (int gid_x = offset_x + skip; gid_x < offset_x + width;            \
             ++gid_x) {                                                        \
          part_result[tid] = REDUCE(part_result[tid], input[gy * stride + gid_x]); \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    if (int missing = height % PPT) {                                          \
      int gid_y = offset_y + end * PPT;                                        \
      tid = -1;                                                                \
                                                                               \
      OPENMP_PRAGMA_TID                                                        \
      for (int m = 0; m < missing; ++m) {                                      \
        if (tid == -1) tid = GET_THREAD_ID;                                    \
        int gy = gid_y + m;                                                    \
        for (int gid_x = offset_x; gid_x < offset_x + width; ++gid_x) {        \
          part_result[tid] = REDUCE(part_result[tid], input[gy * stride + gid_x]); \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (int i = 1; i < num_cores; ++i) {                                      \
      part_result[0] = REDUCE(part_result[0], part_result[i]);                 \
    }                                                                          \
                                                                               \
    return part_result[0];                                                     \
  }

#define BINNING_CPU_2D(NAME, DATA_TYPE, BIN_TYPE, REDUCE, BINNING, PPT)        \
  inline void BINNING##Put(BIN_TYPE *_lmem, uint _offset, uint idx,            \
                           BIN_TYPE val) {                                     \
    if (idx < _offset) {                                                       \
      _lmem[idx] = REDUCE(_lmem[idx], val);                                    \
    }                                                                          \
  }                                                                            \
                                                                               \
  inline std::vector<BIN_TYPE> NAME##Kernel(DATA_TYPE *input, uint num_bins,   \
                                int width, int height, int stride,             \
                                int offset_x = 0, int offset_y = 0) {          \
    int tid = -1;                                                              \
    const int num_cores = GET_NUM_CORES;                                       \
                                                                               \
    std::vector<BIN_TYPE> bins(num_bins);                                      \
    std::vector<BIN_TYPE> lbins(num_cores * num_bins);                         \
                                                                               \
    int end = height / PPT;                                                    \
                                                                               \
    OPENMP_PRAGMA_TID                                                          \
    for (int gid_y = 0; gid_y < end; ++gid_y) {                                \
      int y = offset_y + gid_y * PPT;                                          \
      if (tid == -1) tid = GET_THREAD_ID;                                      \
                                                                               \
      for (int p = 0; p < PPT; ++p) {                                          \
        int gy = y + p;                                                        \
        for (int gid_x = offset_x; gid_x < offset_x + width; ++gid_x) {        \
          BINNING(&lbins[tid * num_bins], num_bins, num_bins, gid_x, gy,       \
                  input[gy * stride + gid_x]);                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    if (int missing = height % PPT) {                                          \
      int gid_y = offset_y + end * PPT;                                        \
      tid = -1;                                                                \
                                                                               \
      OPENMP_PRAGMA_TID                                                        \
      for (int m = 0; m < missing; ++m) {                                      \
        if (tid == -1) tid = GET_THREAD_ID;                                    \
        int gy = gid_y + m;                                                    \
        for (int gid_x = offset_x; gid_x < offset_x + width; ++gid_x) {        \
          BINNING(&lbins[tid * num_bins], num_bins, num_bins, gid_x, gy,       \
                  input[gy * stride + gid_x]);                                 \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    OPENMP_PRAGMA                                                              \
    for (int i = 0; i < num_bins; ++i) {                                       \
      for (int tid = 0; tid < num_cores; ++tid) {                              \
        bins[i] = REDUCE(bins[i], lbins[tid * num_bins + i]);                  \
      }                                                                        \
    }                                                                          \
                                                                               \
    return bins;                                                               \
  }
