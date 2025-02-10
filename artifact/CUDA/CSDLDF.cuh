#pragma once
#include "ScanCommon.cuh"
#include "Utils.cuh"

#define SPLIT_MEMBERS 2
#define FLAG_NOT_READY 0
#define FLAG_READY 0x40000000
#define FLAG_INCLUSIVE 0x80000000
#define FLAG_MASK 0xC0000000
#define VALUE_MASK 0xffff
#define ALL_READY 3
#define MAX_SPIN_COUNT 4

namespace ChainedScanDecoupledFallback {
// lookback, with fallbacks if necessary
template <uint32_t PART_VEC_SIZE, uint32_t WARPS, uint32_t PER_THREAD>
__device__ __forceinline__ void LookbackFallback(const uint32_t partIndex, uint32_t localReduction,
                                                 uint32_t* s_fallback, uint32_t& s_broadcast,
                                                 bool& s_controlFlag, uint32_t* scanIn,
                                                 volatile uint32_t* threadBlockReduction) {
    uint32_t prevRed = 0;
    uint32_t lookbackIndex = (partIndex - 1) * 2;
    while (s_controlFlag) {
        if (threadIdx.x < LANE_COUNT) {
            uint32_t spinCount = 0;
            while (spinCount < MAX_SPIN_COUNT) {
                uint32_t statePayload = threadIdx.x < SPLIT_MEMBERS
                                            ? threadBlockReduction[lookbackIndex + threadIdx.x]
                                            : 0;
                if (__ballot_sync(0xffffffff, (statePayload & FLAG_MASK) > FLAG_NOT_READY) ==
                    ALL_READY) {
                    uint32_t inclBal =
                        __ballot_sync(0xffffffff, (statePayload & FLAG_MASK) == FLAG_INCLUSIVE);
                    if (inclBal) {
                        while (inclBal != ALL_READY) {
                            statePayload = threadIdx.x < SPLIT_MEMBERS
                                               ? threadBlockReduction[lookbackIndex + threadIdx.x]
                                               : 0;
                            inclBal = __ballot_sync(0xffffffff,
                                                    (statePayload & FLAG_MASK) == FLAG_INCLUSIVE);
                        }
                        prevRed += join(statePayload & VALUE_MASK);
                        if (threadIdx.x < SPLIT_MEMBERS) {
                            const uint32_t t = split(prevRed + localReduction) | FLAG_INCLUSIVE;
                            atomicExch(
                                (uint32_t*)&threadBlockReduction[partIndex * 2 + threadIdx.x], t);
                        }
                        if (!threadIdx.x) {
                            s_controlFlag = false;
                            s_broadcast = prevRed;
                        }
                        break;
                    } else {
                        prevRed += join(statePayload & VALUE_MASK);
                        lookbackIndex -= 2;
                        spinCount = 0;
                    }
                } else {
                    spinCount++;
                }
            }

            if (!threadIdx.x && spinCount == MAX_SPIN_COUNT) {
                s_broadcast = lookbackIndex;
            }
        }
        __syncthreads();

        if (s_controlFlag) {
            const uint32_t fallbackIndex = s_broadcast / 2;
            {
                uint32_t t_red = 0;
                #pragma unroll
                for (uint32_t i = threadIdx.x + fallbackIndex * PART_VEC_SIZE, k = 0;
                     k < PER_THREAD; i += blockDim.x, ++k) {
                    uint4 t = reinterpret_cast<uint4*>(scanIn)[i];
                    t_red += t.x + t.y + t.z + t.w;
                }

                const uint32_t s_red = WarpReduceSum(t_red);
                if (!getLaneId()) {
                    s_fallback[WARP_INDEX] = s_red;
                }
            }
            __syncthreads();

            uint32_t f_red = 0;
            if (threadIdx.x < LANE_COUNT) {
                f_red = WarpReduceSum(threadIdx.x < WARPS ? s_fallback[threadIdx.x] : 0);
            }
            __syncthreads();

            if (threadIdx.x < LANE_COUNT) {
                uint32_t f_split = split(f_red) | (fallbackIndex ? FLAG_READY : FLAG_INCLUSIVE);
                uint32_t f_payload = 0;
                if (threadIdx.x < SPLIT_MEMBERS) {
                    f_payload = atomicMax(
                        (uint32_t*)&threadBlockReduction[fallbackIndex * 2 + threadIdx.x], f_split);
                }
                bool inclFound = __ballot_sync(0xffffffff, (f_payload & FLAG_MASK) ==
                                                               FLAG_INCLUSIVE) == ALL_READY;
                if (inclFound) {
                    prevRed += join(f_payload & VALUE_MASK);
                } else {
                    prevRed += f_red;
                }

                if (!fallbackIndex || inclFound) {
                    if (threadIdx.x < SPLIT_MEMBERS) {
                        uint32_t t = split(prevRed + localReduction) | FLAG_INCLUSIVE;
                        atomicExch((uint32_t*)&threadBlockReduction[partIndex * 2 + threadIdx.x],
                                   t);
                    }
                    if (!threadIdx.x) {
                        s_controlFlag = false;
                        s_broadcast = prevRed;
                    }
                } else {
                    lookbackIndex -= 2;
                }
            }
            __syncthreads();
        }
    }
}

template <uint32_t WARPS, uint32_t PER_THREAD>
__device__ __forceinline__ void CSDLDF(
    uint32_t* scanIn, uint32_t* scanOut, volatile uint32_t* threadBlockReduction,
    volatile uint32_t* bump, const uint32_t vectorizedSize,
    void (*ScanVariantFull)(uint4*, uint32_t*, uint32_t*, const uint32_t),
    void (*ScanVariantPartial)(uint4*, uint32_t*, uint32_t*, const uint32_t, const uint32_t)) {
    constexpr uint32_t PART_VEC_SIZE = WARPS * LANE_COUNT * PER_THREAD;
    __shared__ uint32_t s_warpReduction[WARPS];
    __shared__ uint32_t s_fallback[WARPS];
    __shared__ uint32_t s_broadcast;
    __shared__ bool s_controlFlag;

    // Atomically acquire partition index
    if (!threadIdx.x) {
        s_broadcast = atomicAdd((uint32_t*)&bump[0], 1);
        s_controlFlag = true;
    }
    __syncthreads();
    const uint32_t partitionIndex = s_broadcast;

    uint4 tScan[PER_THREAD];
    uint32_t offset = WARP_INDEX * LANE_COUNT * PER_THREAD + partitionIndex * PART_VEC_SIZE;
    if (partitionIndex < gridDim.x - 1) {
        (*ScanVariantFull)(tScan, scanIn, s_warpReduction, offset);
    }

    if (partitionIndex == gridDim.x - 1) {
        (*ScanVariantPartial)(tScan, scanIn, s_warpReduction, offset, vectorizedSize);
    }
    __syncthreads();

    if (threadIdx.x < LANE_COUNT) {
        const bool pred = threadIdx.x < WARPS;
        const uint32_t t = InclusiveWarpScan(pred ? s_warpReduction[threadIdx.x] : 0);
        if (pred) {
            s_warpReduction[threadIdx.x] = t;
        }
    }
    __syncthreads();

    if (threadIdx.x < SPLIT_MEMBERS) {
        atomicExch(
            (uint32_t*)&threadBlockReduction[partitionIndex * 2 + threadIdx.x],
            split(s_warpReduction[WARPS - 1]) | (partitionIndex ? FLAG_READY : FLAG_INCLUSIVE));
    }

    if (partitionIndex) {
        LookbackFallback<PART_VEC_SIZE, WARPS, PER_THREAD>(
            partitionIndex, s_warpReduction[WARPS - 1], s_fallback, s_broadcast, s_controlFlag,
            scanIn, threadBlockReduction);
    }
    __syncthreads();

    const uint32_t prevReduction =
        s_broadcast + (threadIdx.x >= LANE_COUNT ? s_warpReduction[WARP_INDEX - 1] : 0);

    if (partitionIndex < gridDim.x - 1) {
        PropagateFull<PER_THREAD>(tScan, scanOut, prevReduction, offset);
    }

    if (partitionIndex == gridDim.x - 1) {
        PropagatePartial<PER_THREAD>(tScan, scanOut, prevReduction, offset, vectorizedSize);
    }
}

template <uint32_t WARPS, uint32_t PER_THREAD>
__global__ void CSDLDFInclusive(uint32_t* scanIn, uint32_t* scanOut,
                                volatile uint32_t* threadBlockReduction, volatile uint32_t* bump,
                                const uint32_t vectorizedSize) {
    CSDLDF<WARPS, PER_THREAD>(scanIn, scanOut, threadBlockReduction, bump, vectorizedSize,
                              ScanInclusiveFull<PER_THREAD>, ScanInclusivePartial<PER_THREAD>);
}

}

#undef SPLIT_MEMBERS
#undef FLAG_NOT_READY
#undef FLAG_READY
#undef FLAG_INCLUSIVE
#undef FLAG_MASK
#undef VALUE_MASK
#undef ALL_READY
#undef MAX_SPIN_COUNT
