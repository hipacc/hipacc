//
// Copyright (c) 2012, University of Erlangen-Nuremberg
// Copyright (c) 2012, Siemens AG
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <iostream>
#include <vector>

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "hipacc.hpp"

// variables set by Makefile
//#define WIDTH 4096
//#define HEIGHT 4096
#ifndef PPT
#define PPT 1
#endif

using namespace hipacc;
using namespace hipacc::math;


// get time in milliseconds
double time_ms () {
    struct timeval tv;
    gettimeofday (&tv, NULL);

    return ((double)(tv.tv_sec) * 1e+3 + (double)(tv.tv_usec) * 1e-3);
}


// reference implementations
// GOC: global offset correction
void calc_goc(int *in, int *out, int offset, int width, int height, int
        offset_x, int offset_y, int is_width, int is_height) {
    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            out[y*width + x] = in[y*width + x] + offset;
        }
    }
}
void calc_goc(int *in, int *out, int offset, int width, int height) {
    calc_goc(in, out, offset, width, height, 0, 0, width, height);
}
// SAD: sum of absolut differences
int calc_sad(int *in0, int *in1, int *out, int width, int height, int offset_x,
        int offset_y, int is_width, int is_height) {
    int sum = 0;

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            int tmp = abs(in0[y*width + x] - in1[y*width + x]);
            out[y*width + x] = tmp;
            sum += tmp;
        }
    }

    return sum;
}
int calc_sad(int *in0, int *in1, int *out, int width, int height) {
    return calc_sad(in0, in1, out, width, height, 0, 0, width, height);
}
// SSD: sum of square differences
int calc_ssd(int *in0, int *in1, int *out, int width, int height, int offset_x,
        int offset_y, int is_width, int is_height) {
    int sum = 0;

    for (int y=offset_y; y<offset_y+is_height; ++y) {
        for (int x=offset_x; x<offset_x+is_width; ++x) {
            int tmp = (in0[y*width + x] - in1[y*width + x]) *
                (in0[y*width + x] - in1[y*width + x]);
            out[y*width + x] = tmp;
            sum += tmp;
        }
    }

    return sum;
}
int calc_ssd(int *in0, int *in1, int *out, int width, int height) {
    return calc_ssd(in0, in1, out, width, height, 0, 0, width, height);
}


// Kernel description in HIPAcc
class GlobalOffsetCorrection : public Kernel<int> {
    private:
        Accessor<int> &input;
        int offset;

    public:
        GlobalOffsetCorrection(IterationSpace<int> &iter, Accessor<int> &input,
                int offset) :
            Kernel(iter),
            input(input),
            offset(offset)
        { addAccessor(&input); }

        void kernel() {
            output() = input() + offset;
        }
};
class AbsoluteDifferences : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;

    public:
        AbsoluteDifferences(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1) :
            Kernel(iter),
            input0(input0),
            input1(input1)
        {
            addAccessor(&input0);
            addAccessor(&input1);
        }

        void kernel() {
            output() = abs(input0()-input1());
        }
};
class SquareDifferences : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;

    public:
        SquareDifferences(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1) :
            Kernel(iter),
            input0(input0),
            input1(input1)
        {
            addAccessor(&input0);
            addAccessor(&input1);
        }

        void kernel() {
            output() = (input0()-input1())*(input0()-input1());
        }
};
class Read1 : public Kernel<int> {
    private:
        Accessor<int> &input0;

    public:
        Read1(IterationSpace<int> &iter, Accessor<int> &input0) :
            Kernel(iter),
            input0(input0)
        { addAccessor(&input0); }

        void kernel() {
            output() = input0();
        }
};
class Read2 : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;

    public:
        Read2(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1) :
            Kernel(iter),
            input0(input0),
            input1(input1)
        {
            addAccessor(&input0);
            addAccessor(&input1);
        }

        void kernel() {
            output() = input0() + input1();
        }
};
class Read3 : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;
        Accessor<int> &input2;

    public:
        Read3(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1, Accessor<int> &input2) :
            Kernel(iter),
            input0(input0),
            input1(input1),
            input2(input2)
        {
            addAccessor(&input0);
            addAccessor(&input1);
            addAccessor(&input2);
        }

        void kernel() {
            output() = input0() + input1() + input2();
        }
};
class Read4 : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;
        Accessor<int> &input2;
        Accessor<int> &input3;

    public:
        Read4(IterationSpace<int> &iter, Accessor<int> &input0, Accessor<int>
                &input1, Accessor<int> &input2, Accessor<int> &input3) :
            Kernel(iter),
            input0(input0),
            input1(input1),
            input2(input2),
            input3(input3)
        {
            addAccessor(&input0);
            addAccessor(&input1);
            addAccessor(&input2);
            addAccessor(&input3);
        }

        void kernel() {
            output() = input0() + input1() + input2() + input3();
        }
};
class Read5 : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;
        Accessor<int> &input2;
        Accessor<int> &input3;
        Accessor<int> &input4;

    public:
        Read5(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1, Accessor<int> &input2,
                Accessor<int> &input3, Accessor<int> &input4) :
            Kernel(iter),
            input0(input0),
            input1(input1),
            input2(input2),
            input3(input3),
            input4(input4)
        {
            addAccessor(&input0);
            addAccessor(&input1);
            addAccessor(&input2);
            addAccessor(&input3);
            addAccessor(&input4);
        }

        void kernel() {
            output() = input0() + input1() + input2() + input3() + input4();
        }
};
class Read6 : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;
        Accessor<int> &input2;
        Accessor<int> &input3;
        Accessor<int> &input4;
        Accessor<int> &input5;

    public:
        Read6(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1, Accessor<int> &input2,
                Accessor<int> &input3, Accessor<int> &input4,
                Accessor<int> &input5) :
            Kernel(iter),
            input0(input0),
            input1(input1),
            input2(input2),
            input3(input3),
            input4(input4),
            input5(input5)
        {
            addAccessor(&input0);
            addAccessor(&input1);
            addAccessor(&input2);
            addAccessor(&input3);
            addAccessor(&input4);
            addAccessor(&input5);
        }

        void kernel() {
            output() = input0() + input1() + input2() + input3() + input4() + input5();
        }
};
class Read7 : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;
        Accessor<int> &input2;
        Accessor<int> &input3;
        Accessor<int> &input4;
        Accessor<int> &input5;
        Accessor<int> &input6;

    public:
        Read7(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1, Accessor<int> &input2,
                Accessor<int> &input3, Accessor<int> &input4,
                Accessor<int> &input5, Accessor<int> &input6) :
            Kernel(iter),
            input0(input0),
            input1(input1),
            input2(input2),
            input3(input3),
            input4(input4),
            input5(input5),
            input6(input6)
        {
            addAccessor(&input0);
            addAccessor(&input1);
            addAccessor(&input2);
            addAccessor(&input3);
            addAccessor(&input4);
            addAccessor(&input5);
            addAccessor(&input6);
        }

        void kernel() {
            output() = input0() + input1() + input2() + input3() + input4() + input5() + input6();
        }
};
class Read8 : public Kernel<int> {
    private:
        Accessor<int> &input0;
        Accessor<int> &input1;
        Accessor<int> &input2;
        Accessor<int> &input3;
        Accessor<int> &input4;
        Accessor<int> &input5;
        Accessor<int> &input6;
        Accessor<int> &input7;

    public:
        Read8(IterationSpace<int> &iter, Accessor<int> &input0,
                Accessor<int> &input1, Accessor<int> &input2,
                Accessor<int> &input3, Accessor<int> &input4,
                Accessor<int> &input5, Accessor<int> &input6,
                Accessor<int> &input7) :
            Kernel(iter),
            input0(input0),
            input1(input1),
            input2(input2),
            input3(input3),
            input4(input4),
            input5(input5),
            input6(input6),
            input7(input7)
        {
            addAccessor(&input0);
            addAccessor(&input1);
            addAccessor(&input2);
            addAccessor(&input3);
            addAccessor(&input4);
            addAccessor(&input5);
            addAccessor(&input6);
            addAccessor(&input7);
        }

        void kernel() {
            output() = input0() + input1() + input2() + input3() + input4() + input5() + input6() + input7();
        }
};


int main(int argc, const char **argv) {
    double time0, time1, dt;
    const int width = WIDTH;
    const int height = HEIGHT;
    int offset = 5;
    std::vector<float> timings;
    float timing = 0.0f;

    // host memory for image of width x height pixels
    int *input0 = (int *)malloc(sizeof(int)*width*height);
    int *input1 = (int *)malloc(sizeof(int)*width*height);
    int *reference_in0 = (int *)malloc(sizeof(int)*width*height);
    int *reference_in1 = (int *)malloc(sizeof(int)*width*height);
    int *reference_out0 = (int *)malloc(sizeof(int)*width*height);
    int *reference_out1 = (int *)malloc(sizeof(int)*width*height);
    int *reference_out2 = (int *)malloc(sizeof(int)*width*height);

    // input and output image of width x height pixels
    Image<int> IN0(width, height);
    Image<int> IN1(width, height);
    Image<int> IN2(width, height);
    Image<int> IN3(width, height);
    Image<int> IN4(width, height);
    Image<int> IN5(width, height);
    Image<int> IN6(width, height);
    Image<int> IN7(width, height);
    Image<int> OUT0(width, height);
    Image<int> OUT1(width, height);
    Image<int> OUT2(width, height);

    Accessor<int> AccIn0(IN0);
    Accessor<int> AccIn1(IN1);
    Accessor<int> AccIn2(IN2);
    Accessor<int> AccIn3(IN3);
    Accessor<int> AccIn4(IN4);
    Accessor<int> AccIn5(IN5);
    Accessor<int> AccIn6(IN6);
    Accessor<int> AccIn7(IN7);

    // initialize data
    #define DELTA 0.001f
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            input0[y*width + x] = (int) (x*height + y) * DELTA;
            input1[y*width + x] = (int) (y*width + x) * DELTA;
            reference_in0[y*width + x] = (int) (x*height + y) * DELTA;
            reference_in1[y*width + x] = (int) (y*width + x) * DELTA;
            reference_out0[y*width + x] = (int) (3.12451);
            reference_out1[y*width + x] = (int) (3.12451);
            reference_out2[y*width + x] = (int) (3.12451);
        }
    }

    IterationSpace<int> ISOut0(OUT0);
    IterationSpace<int> ISOut1(OUT1);
    IterationSpace<int> ISOut2(OUT2);
    GlobalOffsetCorrection GOC(ISOut0, AccIn0, offset);
    AbsoluteDifferences AD(ISOut1, AccIn0, AccIn1);
    SquareDifferences SD(ISOut2, AccIn0, AccIn1);
    Read1 R1(ISOut0, AccIn0);
    Read2 R2(ISOut0, AccIn0, AccIn1);
    Read3 R3(ISOut0, AccIn0, AccIn1, AccIn2);
    Read4 R4(ISOut0, AccIn0, AccIn1, AccIn2, AccIn3);
    Read5 R5(ISOut0, AccIn0, AccIn1, AccIn2, AccIn3, AccIn4);
    Read6 R6(ISOut0, AccIn0, AccIn1, AccIn2, AccIn3, AccIn4, AccIn5);
    Read7 R7(ISOut0, AccIn0, AccIn1, AccIn2, AccIn3, AccIn4, AccIn5, AccIn6);
    Read8 R8(ISOut0, AccIn0, AccIn1, AccIn2, AccIn3, AccIn4, AccIn5, AccIn6, AccIn7);

    IN0 = input0;
    IN1 = input1;
    IN2 = input1;
    IN3 = input1;
    IN4 = input1;
    IN5 = input1;
    IN6 = input1;
    IN7 = input1;

    // warmup
    R1.execute();
    R2.execute();
    R3.execute();
    R4.execute();
    R5.execute();
    R6.execute();
    R7.execute();
    R8.execute();

    GOC.execute();
    AD.execute();
    SD.execute();

    fprintf(stderr, "Calculating 1 image kernel ...\n");
    R1.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);
    size_t memory_size = sizeof(int)*width*height;
    float bandwidth_MBs = (2.0f * (double)memory_size)/(timing/1000 * (double)(1 << 20));


    fprintf(stderr, "Calculating 2 image kernel ...\n");
    R2.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating 3 image kernel ...\n");
    R3.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating 4 image kernel ...\n");
    R4.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating 5 image kernel ...\n");
    R5.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating 6 image kernel ...\n");
    R6.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating 7 image kernel ...\n");
    R7.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating 8 image kernel ...\n");
    R8.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating global offset correction kernel ...\n");
    GOC.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating absolute difference kernel ...\n");
    AD.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    fprintf(stderr, "Calculating square difference kernel ...\n");
    SD.execute();
    timing = hipaccGetLastKernelTiming();
    timings.push_back(timing);
    fprintf(stderr, "Hipacc: %.3f ms, %.3f Mpixel/s\n", timing, (width*height/timing)/1000);


    // print statistics
    fprintf(stderr, "PPT: %d", PPT);
    for (std::vector<float>::const_iterator it = timings.begin();
         it != timings.end(); ++it) {
        fprintf(stderr, "\t%.3f", *it);
    }
    fprintf(stderr, "\n\n");

    // print achieved bandwidth
    fprintf(stderr, "Bandwidth for memory size [MB]: %lu\n", memory_size/(1024*1024));
    fprintf(stderr, "Bandwidth [MB/s]: %f\n", bandwidth_MBs);
    fprintf(stderr, "Bandwidth [GB/s]: %f\n", bandwidth_MBs/1024);
    fprintf(stderr, "\n\n");


    // get pointer to result data
    int *output0 = OUT0.getData();
    int *output1 = OUT1.getData();
    int *output2 = OUT2.getData();


    // GOC
    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    calc_goc(reference_in0, reference_out0, offset, width, height);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (width*height/dt)/1000);

    // SAD
    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    calc_sad(reference_in0, reference_in1, reference_out1, width, height);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (width*height/dt)/1000);

    // SSD
    fprintf(stderr, "\nCalculating reference ...\n");
    time0 = time_ms();

    calc_ssd(reference_in0, reference_in1, reference_out2, width, height);

    time1 = time_ms();
    dt = time1 - time0;
    fprintf(stderr, "Reference: %.3f ms, %.3f Mpixel/s\n", dt, (width*height/dt)/1000);

    // compare results
    fprintf(stderr, "\nComparing results for GOC ... ");
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            if (reference_out0[y*width + x] != output0[y*width + x]) {
                fprintf(stderr, " FAILED, at (%d,%d): %d vs. %d\n", x, y,
                        reference_out0[y*width + x], output0[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    fprintf(stderr, "PASSED\n");
    fprintf(stderr, "Comparing results for AD ... ");
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            if (reference_out1[y*width + x] != output1[y*width + x]) {
                fprintf(stderr, " FAILED, at (%d,%d): %d vs. %d\n", x, y,
                        reference_out1[y*width + x], output1[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    fprintf(stderr, "PASSED\n");
    fprintf(stderr, "Comparing results for SD ... ");
    for (int y=0; y<height; y++) {
        for (int x=0; x<width; x++) {
            if (reference_out2[y*width + x] != output2[y*width + x]) {
                fprintf(stderr, " FAILED, at (%d,%d): %d vs. %d\n", x, y,
                        reference_out2[y*width + x], output2[y*width + x]);
                exit(EXIT_FAILURE);
            }
        }
    }
    fprintf(stderr, "PASSED\n");
    fprintf(stderr, "All Tests PASSED\n");

    // memory cleanup
    free(input0);
    free(input1);
    free(reference_in0);
    free(reference_in1);
    free(reference_out0);
    free(reference_out1);
    free(reference_out2);

    return EXIT_SUCCESS;
}

