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
    csvFile << "time\n";  // CSV header
    for (float val : data) {
        csvFile << val << "\n";
    }
    csvFile.close();
    std::cout << "CSV data written to " << filename << std::endl;
}

class CubTestRunner {
public:
    // Run the test for a given size and batch count.
    // Returns a vector of individual iteration times (in ms), excluding the warmup.
    std::vector<float> runTest(uint32_t size, uint32_t batchCount,
                               uint32_t* scanIn, uint32_t* scanOut, uint32_t* err) {
        std::vector<float> times(batchCount);
        printf("Running CUB test for size %u, batch count %u\n", size, batchCount);

        // Warm-up: determine temporary storage size.
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, scanIn, scanOut, size);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i <= batchCount; ++i) {
            InitOne<<<256, 256>>>(scanIn, size);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, scanIn, scanOut, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis = 0.0f;
            cudaEventElapsedTime(&millis, start, stop);

            // Discard the first (warmup) iteration.
            if (i > 0) {
                testsPassed += DispatchValidateInclusive(scanOut, err, size) ? 1 : 0;
                totalTime += millis;
                times[i - 1] = millis;  // Record individual time in milliseconds.
            }
        }

        float avgTime = totalTime / batchCount;
        float throughput = size / (avgTime / 1000.0f);  // keys/sec calculation.
        printf("CUB: Size %u: Average time: %f ms, Throughput: %E keys/sec, Tests passed: %u/%u\n",
               size, avgTime, throughput, testsPassed, batchCount);

        cudaDeviceSynchronize();
        cudaFree(d_temp_storage);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return times;
    }
};

class CsdldfTestRunner {
public:
    std::vector<float> runTest(uint32_t size, uint32_t batchCount,
                               uint32_t* scanIn, uint32_t* scanOut, uint32_t* err) {
        std::vector<float> times(batchCount);
        printf("Running CSDLDF test for size %u, batch count %u\n", size, batchCount);

        const uint32_t k_csdldfThreads = WARPS * LANE_COUNT;
        const uint32_t k_partitionSize = k_csdldfThreads * UINT4_PER_THREAD * 4;
        const uint32_t k_vecSize = size / 4;  // Assumes size is a multiple of 4.

        uint32_t* index;
        uint32_t* threadBlockReduction;
        const uint32_t threadBlocks = (size + k_partitionSize - 1) / k_partitionSize;
        cudaMalloc(&index, sizeof(uint32_t));
        cudaMalloc(&threadBlockReduction, threadBlocks * 2 * sizeof(uint32_t));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float totalTime = 0.0f;
        uint32_t testsPassed = 0;
        for (uint32_t i = 0; i <= batchCount; ++i) {
            InitOne<<<256, 256>>>(scanIn, size);
            cudaDeviceSynchronize();

            cudaEventRecord(start);
            cudaMemset(index, 0, sizeof(uint32_t));
            cudaMemset(threadBlockReduction, 0, threadBlocks * 2 * sizeof(uint32_t));
            ChainedScanDecoupledFallback::CSDLDFInclusive<WARPS, UINT4_PER_THREAD>
                <<<threadBlocks, k_csdldfThreads>>>(scanIn, scanOut, threadBlockReduction, index, k_vecSize);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis = 0.0f;
            cudaEventElapsedTime(&millis, start, stop);

            if (i > 0) {
                testsPassed += DispatchValidateInclusive(scanOut, err, size) ? 1 : 0;
                totalTime += millis;
                times[i - 1] = millis;
            }
        }

        float avgTime = totalTime / batchCount;
        float throughput = size / (avgTime / 1000.0f);
        printf("CSDLDF: Size %u: Average time: %f ms, Throughput: %E keys/sec, Tests passed: %u/%u\n",
               size, avgTime, throughput, testsPassed, batchCount);

        cudaDeviceSynchronize();
        cudaFree(index);
        cudaFree(threadBlockReduction);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return times;
    }
};

class MemcpyTestRunner {
public:
    // Note: The memcpy test does not require an error buffer.
    std::vector<float> runTest(uint32_t size, uint32_t batchCount,
                               uint32_t* scanIn, uint32_t* scanOut) {
        std::vector<float> times(batchCount);
        printf("Running Memcpy test for size %u, batch count %u\n", size, batchCount);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        for (uint32_t i = 0; i <= batchCount; ++i) {
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            cudaMemcpy(scanIn, scanOut, size * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float millis = 0.0f;
            cudaEventElapsedTime(&millis, start, stop);

            if (i > 0) {  // Skip warmup iteration.
                times[i - 1] = millis;
            }
        }

        float totalTime = 0.0f;
        for (float t : times)
            totalTime += t;
        float avgTime = totalTime / batchCount;
        float throughput = size / (avgTime / 1000.0f);
        printf("Memcpy: Size %u: Average time: %f ms, Throughput: %E keys/sec\n",
               size, avgTime, throughput);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return times;
    }
};

template <typename TestRunner>
std::vector<std::pair<uint32_t, std::vector<float>>> RunCudaTests(
    TestRunner& runner,
    uint32_t batchCount,
    uint32_t minPow,
    uint32_t maxPow,
    uint32_t* scanIn,
    uint32_t* scanOut,
    uint32_t* err) {
    std::vector<std::pair<uint32_t, std::vector<float>>> results;
    for (uint32_t i = minPow; i <= maxPow; ++i) {
        uint32_t n = 1u << i;
        std::vector<float> times = runner.runTest(n, batchCount, scanIn, scanOut, err);
        results.push_back(std::make_pair(n, times));
    }
    return results;
}

std::vector<std::pair<uint32_t, std::vector<float>>> RunCudaTestsMemcpy(
    MemcpyTestRunner& runner,
    uint32_t batchCount,
    uint32_t minPow,
    uint32_t maxPow,
    uint32_t* scanIn,
    uint32_t* scanOut) {
    std::vector<std::pair<uint32_t, std::vector<float>>> results;
    for (uint32_t i = minPow; i <= maxPow; ++i) {
        uint32_t n = 1u << i;
        std::vector<float> times = runner.runTest(n, batchCount, scanIn, scanOut);
        results.push_back(std::make_pair(n, times));
    }
    return results;
}

int main(int argc, char* argv[]) {
    // Allowed usage: <testType> [record [csvName]]
    // testType must be one of "cub", "csdldf", or "memcpy".
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <cub|csdldf|memcpy> [record [csvName]]\n";
        return EXIT_FAILURE;
    }
    std::string testType = argv[1];
    bool record = false;
    std::string csvName = "";
    if (argc >= 3) {
        if (std::string(argv[2]) == "record") {
            record = true;
            if (argc == 4) {
                csvName = argv[3];
            }
        } else {
            std::cerr << "Unknown second argument: " << argv[2] << std::endl;
            return EXIT_FAILURE;
        }
    }

    constexpr uint32_t testBatchSize = 2000;
    constexpr uint32_t minPow = 10;
    constexpr uint32_t maxPow = 25;

    uint32_t* err = nullptr;
    uint32_t* scan_in = nullptr;
    uint32_t* scan_out = nullptr;
    cudaMalloc(&err, sizeof(uint32_t));
    cudaMalloc(&scan_in, (1u << maxPow) * sizeof(uint32_t));
    cudaMalloc(&scan_out, (1u << maxPow) * sizeof(uint32_t));

    cudaError_t errStatus = cudaGetLastError();
    if (errStatus != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(errStatus) << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::pair<uint32_t, std::vector<float>>> results;
    if (testType == "cub") {
        CubTestRunner runner;
        results = RunCudaTests(runner, testBatchSize, minPow, maxPow, scan_in, scan_out, err);
    } else if (testType == "csdldf") {
        CsdldfTestRunner runner;
        results = RunCudaTests(runner, testBatchSize, minPow, maxPow, scan_in, scan_out, err);
    } else if (testType == "memcpy") {
        MemcpyTestRunner runner;
        results = RunCudaTestsMemcpy(runner, testBatchSize, minPow, maxPow, scan_in, scan_out);
    } else {
        std::cerr << "Unknown test type: " << testType << std::endl;
        return EXIT_FAILURE;
    }

    if (record) {
        for (const auto& result : results) {
            uint32_t size = result.first;
            const std::vector<float>& times = result.second;
            std::string filename = csvName + "_" + testType + "_Size_" + std::to_string(size) + ".csv";
            WriteVectorToCSV(times, filename);
        }
    }

    cudaFree(err);
    cudaFree(scan_in);
    cudaFree(scan_out);
    return EXIT_SUCCESS;
}
