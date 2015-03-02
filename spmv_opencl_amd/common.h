#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <string>
#include <math.h>

#include <fstream>
#include <iostream>

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/opencl.h"
#endif

#define BHSPARSE_SUCCESS 0
#define BHSPARSE_UNSUPPORTED_DEVICE -2

using namespace std;

#ifndef USE_DOUBLE
#define USE_DOUBLE 1
#endif

#if USE_DOUBLE
typedef double   value_type;
#else
typedef float   value_type;
#endif

#define NUM_RUN 200

#ifndef USE_SVM_ALWAYS
#define USE_SVM_ALWAYS 1
#endif

// (SEG_H * THREADBUNCH) should be multiply of 16
// STEP should be less than THREADBUNCH

#define THREADBUNCH  64
#define THREADGROUP  256

#define SEG_H        8
#define STEP         15

// timing functions
#include <sys/time.h>

#include <sys/types.h>
#include <dirent.h>

struct bhsparse_timer {
    timeval t1, t2;
    struct timezone tzone;

    void start() {
        gettimeofday(&t1, &tzone);
    }

    double stop() {
        gettimeofday(&t2, &tzone);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};


#endif // COMMON_H
