#pragma once
#include <cstdint>
#include <cstdio>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define LANE_COUNT 32                         
#define LANE_MASK 31                          
#define LANE_LOG 5                            
#define WARP_INDEX (threadIdx.x >> LANE_LOG)  

__device__ __forceinline__ uint32_t getLaneId() {
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return laneId;
}

//Warp scans
__device__ __forceinline__ uint32_t InclusiveWarpScan(uint32_t val) {
    uint32_t t = __shfl_up_sync(0xffffffff, val, 1, LANE_COUNT);
    if (getLaneId() >= 1) val += t;
    t = __shfl_up_sync(0xffffffff, val, 2, LANE_COUNT);
    if (getLaneId() >= 2) val += t;
    t = __shfl_up_sync(0xffffffff, val, 4, LANE_COUNT);
    if (getLaneId() >= 4) val += t;
    t = __shfl_up_sync(0xffffffff, val, 8, LANE_COUNT);
    if (getLaneId() >= 8) val += t;
    t = __shfl_up_sync(0xffffffff, val, 16, LANE_COUNT);
    if (getLaneId() >= 16) val += t;
    return val;
}

__device__ __forceinline__ uint32_t InclusiveWarpScanCircularShift(uint32_t val) {
    uint32_t t = __shfl_up_sync(0xffffffff, val, 1, LANE_COUNT);
    if (getLaneId() >= 1) val += t;
    t = __shfl_up_sync(0xffffffff, val, 2, LANE_COUNT);
    if (getLaneId() >= 2) val += t;
    t = __shfl_up_sync(0xffffffff, val, 4, LANE_COUNT);
    if (getLaneId() >= 4) val += t;
    t = __shfl_up_sync(0xffffffff, val, 8, LANE_COUNT);
    if (getLaneId() >= 8) val += t;
    t = __shfl_up_sync(0xffffffff, val, 16, LANE_COUNT);
    if (getLaneId() >= 16) val += t;
    return __shfl_sync(0xffffffff, val, getLaneId() + LANE_MASK & LANE_MASK);
}

__device__ __forceinline__ uint32_t WarpReduceSum(uint32_t val) {
    val += __shfl_xor_sync(0xffffffff, val, 16, LANE_COUNT);
    val += __shfl_xor_sync(0xffffffff, val, 8, LANE_COUNT);
    val += __shfl_xor_sync(0xffffffff, val, 4, LANE_COUNT);
    val += __shfl_xor_sync(0xffffffff, val, 2, LANE_COUNT);
    val += __shfl_xor_sync(0xffffffff, val, 1, LANE_COUNT);
    return val;
}

__device__ __forceinline__ uint4 SetXAddYZW(uint32_t valToAdd, uint4 val) {
    return make_uint4(valToAdd, val.y + valToAdd, val.z + valToAdd, val.w + valToAdd);
}

__device__ __forceinline__ uint4 AddUintToUint4(uint32_t valToAdd, uint4 val) {
    return make_uint4(val.x + valToAdd, val.y + valToAdd, val.z + valToAdd, val.w + valToAdd);
}

__device__ __forceinline__ uint32_t ReduceUint4(uint4 val) {
    return val.x + val.y + val.z + val.w;
}

__device__ __forceinline__ uint32_t join(uint32_t val) {
    const uint32_t x = threadIdx.x ^ 1;
    return val << 16 * threadIdx.x | __shfl_sync(0xffffffff, val, x) << 16 * x;
}

__device__ __forceinline__ uint32_t split(uint32_t val) {
    return val >> threadIdx.x * 16u & 0xffff;
}