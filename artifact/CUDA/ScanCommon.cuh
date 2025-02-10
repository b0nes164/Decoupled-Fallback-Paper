#pragma once
#include "Utils.cuh"
template <uint32_t PER_THREAD>
__device__ __forceinline__ void ScanInclusiveFull(uint4* tScan, uint32_t* scan,
                                                  uint32_t* s_warpReduction,
                                                  const uint32_t offset) {
    uint32_t warpReduction = 0;
    #pragma unroll
    for (uint32_t i = getLaneId() + offset, k = 0; k < PER_THREAD; i += LANE_COUNT, ++k) {
        tScan[k] = reinterpret_cast<uint4*>(scan)[i];
        tScan[k].y += tScan[k].x;
        tScan[k].z += tScan[k].y;
        tScan[k].w += tScan[k].z;

        const uint32_t t = InclusiveWarpScanCircularShift((getLaneId() ? 0 : warpReduction) + tScan[k].w);
        tScan[k] = AddUintToUint4((getLaneId() ? t : warpReduction), tScan[k]);
        warpReduction = t;
    }

    if (!getLaneId())
        s_warpReduction[WARP_INDEX] = warpReduction;
}

template <uint32_t PER_THREAD>
__device__ __forceinline__ void ScanInclusivePartial(uint4* tScan, uint32_t* scan,
                                                     uint32_t* s_warpReduction,
                                                     const uint32_t offset,
                                                     const uint32_t vecSize) {
    uint32_t warpReduction = 0;
    #pragma unroll
    for (uint32_t i = getLaneId() + offset, k = 0; k < PER_THREAD; i += LANE_COUNT, ++k) {
        tScan[k] = i < vecSize ? reinterpret_cast<uint4*>(scan)[i] : make_uint4(0, 0, 0, 0);
        tScan[k].y += tScan[k].x;
        tScan[k].z += tScan[k].y;
        tScan[k].w += tScan[k].z;

        const uint32_t t = InclusiveWarpScanCircularShift((getLaneId() ? 0 : warpReduction) + tScan[k].w);
        tScan[k] = AddUintToUint4((getLaneId() ? t : warpReduction), tScan[k]);
        warpReduction = t;
    }

    if (!getLaneId())
        s_warpReduction[WARP_INDEX] = warpReduction;
}

template <uint32_t PER_THREAD>
__device__ __forceinline__ void PropagateFull(uint4* tScan, uint32_t* scan,
                                              const uint32_t prevReduction, const uint32_t offset) {
    #pragma unroll
    for (uint32_t i = getLaneId() + offset, k = 0; k < PER_THREAD; i += LANE_COUNT, ++k) {
        reinterpret_cast<uint4*>(scan)[i] = AddUintToUint4(prevReduction, tScan[k]);
    }
}

template <uint32_t PER_THREAD>
__device__ __forceinline__ void PropagatePartial(uint4* tScan, uint32_t* scan,
                                                 const uint32_t prevReduction,
                                                 const uint32_t offset, const uint32_t vecSize) {
    #pragma unroll
    for (uint32_t i = getLaneId() + offset, k = 0; k < PER_THREAD; i += LANE_COUNT, ++k) {
        if (i < vecSize) {
            reinterpret_cast<uint4*>(scan)[i] = AddUintToUint4(prevReduction, tScan[k]);
        }
    }
}