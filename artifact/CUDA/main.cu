#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "CSDLDF.cuh"

#define WARPS 8
#define UINT4_PER_THREAD 4
#define checkCudaError()                                         \
    {                                                            \
        cudaError_t err = cudaGetLastError();                    \
        if (err != cudaSuccess) {                                \
            printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        }                                                        \
    }

__global__ void InitOne(uint32_t* scan, uint32_t size) {
    const uint32_t increment = blockDim.x * gridDim.x;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += increment) {
        scan[i] = 1;
    }
}

__global__ void ValidateInclusive(uint32_t* scan, uint32_t* errCount, uint32_t size) {
    const uint32_t increment = blockDim.x * gridDim.x;
    for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += increment) {
        if (scan[i] != i + 1) {
            atomicAdd(&errCount[0], 1);
        }
    }
}

bool DispatchValidateInclusive(uint32_t* scanOut, uint32_t* err, uint32_t size) {
    uint32_t errCount[1] = {0xffffffff};
    cudaMemset(err, 0, sizeof(uint32_t));
    ValidateInclusive<<<256, 256>>>(scanOut, err, size);
    cudaMemcpy(&errCount, err, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return !errCount[0];
}

void WriteVectorToCSV(const std::vector<float>& data, const std::string& filename) {
    std::ofstream csvFile(filename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    csvFile << "throughput" << std::endl;
    for (size_t i = 0; i < data.size(); ++i) {
        csvFile << data[i] << std::endl;
    }
    csvFile.close();
    std::cout << "CSV data written to " << filename << std::endl;
}

void BatchTimingCubChainedScan(uint32_t size, uint32_t batchCount, uint32_t* scanIn,
                               uint32_t* scanOut, uint32_t* err, bool shouldRecord,
                               std::string csvName) {
    std::vector<float> time;
    if (shouldRecord) {
        time.resize(batchCount);
    }

    printf("Beginning CUB inclusive batch timing test at:\n");
    printf("Size: %u\n", size);
    printf("Test batch size: %u\n", batchCount);

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, scanIn, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t testsPassed = 0;
    float totalTime = 0.0f;
    for (uint32_t i = 0; i <= batchCount; ++i) {
        InitOne<<<256, 256>>>(scanIn, size);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, scanIn, scanOut, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float millis;
        cudaEventElapsedTime(&millis, start, stop);
        // Discard the first iteration (warmup)
        if (i > 0) {
            testsPassed += DispatchValidateInclusive(scanOut, err, size) ? 1 : 0;
            totalTime += millis;
            if (shouldRecord) {
                time[i - 1] = size / (millis / 1000.0f);
            }
        }
    }

    printf("\n");
    totalTime /= 1000.0f;
    printf("TOTAL TESTS PASSED: %u / %u\n", testsPassed, batchCount);
    printf("Total time elapsed: %f seconds\n", totalTime);
    printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size,
           size / totalTime * batchCount);

    if (shouldRecord) {
        WriteVectorToCSV(time, csvName + "_" + std::to_string(size) + ".csv");
    }

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_temp_storage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void BatchTimingCSDLDF(uint32_t size, uint32_t batchCount, uint32_t* scanIn, uint32_t* scanOut,
                       uint32_t* err, bool shouldRecord, std::string csvName) {
    std::vector<float> time;
    if (shouldRecord) {
        time.resize(batchCount);
    }

    printf("Beginning DecoupledFallback inclusive batch timing test at:\n");
    printf("Size: %u\n", size);
    printf("Test batch size: %u\n", batchCount);

    const uint32_t k_csdldfThreads = WARPS * LANE_COUNT;
    const uint32_t k_partitionSize = k_csdldfThreads * UINT4_PER_THREAD * 4;
    const uint32_t k_vecSize = size / 4;  // Assume multiple of 4

    uint32_t* index;
    uint32_t* threadBlockReduction;
    const uint32_t threadBlocks = (size + k_partitionSize - 1) / k_partitionSize;
    cudaMalloc(&index, sizeof(uint32_t));
    cudaMalloc(&threadBlockReduction, threadBlocks * 2 * sizeof(uint32_t));

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    uint32_t testsPassed = 0;
    float totalTime = 0.0f;
    for (uint32_t i = 0; i <= batchCount; ++i) {
        InitOne<<<256, 256>>>(scanIn, size);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cudaMemset(index, 0, sizeof(uint32_t));
        cudaMemset(threadBlockReduction, 0, threadBlocks * 2 * sizeof(uint32_t));
        ChainedScanDecoupledFallback::CSDLDFInclusive<WARPS, UINT4_PER_THREAD>
            <<<threadBlocks, k_csdldfThreads>>>(scanIn, scanOut, threadBlockReduction, index,
                                                k_vecSize);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float millis;
        cudaEventElapsedTime(&millis, start, stop);
        // Discard the first iteration (warmup)
        if (i > 0) {
            testsPassed += DispatchValidateInclusive(scanOut, err, size) ? 1 : 0;
            totalTime += millis;
            if (shouldRecord) {
                time[i - 1] = size / (millis / 1000.0f);
            }
        }
    }

    printf("\n");
    totalTime /= 1000.0f;
    printf("TOTAL TESTS PASSED: %u / %u\n", testsPassed, batchCount);
    printf("Total time elapsed: %f seconds\n", totalTime);
    printf("Estimated speed at %u 32-bit elements: %E keys/sec\n\n", size,
           size / totalTime * batchCount);

    if (shouldRecord) {
        WriteVectorToCSV(time, csvName + "_" + std::to_string(size) + ".csv");
    }

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(index);
    cudaFree(threadBlockReduction);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    bool record = false;
    std::string csvName = "";
    std::string testType;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cub|csdl|csdldf> [record [csvName]]\n";
        return EXIT_FAILURE;
    }

    testType = argv[1];
    if (testType != "cub" && testType != "csdldf") {
        std::cerr << "Unknown test type: " << testType << "\n";
        return EXIT_FAILURE;
    }

    if (argc > 2) {
        std::string arg2(argv[2]);
        if (arg2 == "record") {
            record = true;
            if (argc > 3) {
                csvName = argv[3];
            }
        }
    }

    constexpr uint32_t testBatchSize = 500;
    constexpr uint32_t maxPowTwo = 25;

    uint32_t* err = nullptr;
    uint32_t* scan_in = nullptr;
    uint32_t* scan_out = nullptr;
    cudaMalloc(&err, sizeof(uint32_t));
    cudaMalloc(&scan_in, (1 << maxPowTwo) * sizeof(uint32_t));
    cudaMalloc(&scan_out, (1 << maxPowTwo) * sizeof(uint32_t));
    checkCudaError();

    //Only do one test at a time
    //We want the tests to be cold, otherwise cacheing introduces data artifacts
    for (uint32_t i = 10; i <= maxPowTwo; ++i) {
        uint32_t n = 1 << i;
        if (testType == "cub") {
            BatchTimingCubChainedScan(n, testBatchSize, scan_in, scan_out, err, record,
                                      csvName + "_CUB");
        } else if (testType == "csdldf") {
            BatchTimingCSDLDF(n, testBatchSize, scan_in, scan_out, err, record,
                              csvName + "_CSDLDF");
        }
    }
    
    checkCudaError();
    cudaFree(err);
    cudaFree(scan_in);
    cudaFree(scan_out);
    return EXIT_SUCCESS;
}
