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
#define USE_DOUBLE 0
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

#define THREADBUNCH  8
#define THREADGROUP  32

#define SEG_H        16
#define STEP         6

// timing functions
// see http://stackoverflow.com/questions/10905892/equivalent-of-gettimeday-for-windows
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 

typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}

struct bhsparse_timer {
    timeval t1, t2;

    void start() {
        gettimeofday(&t1);
    }

    double stop() {
        gettimeofday(&t2);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};


#endif // COMMON_H
