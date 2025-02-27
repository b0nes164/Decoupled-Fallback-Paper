# SPAA2025 Decoupled Fallback Artifact
This repository contains the LaTeX file and artifact the SPAA2025 submission *Decoupled Fallback: A Portable Single-Pass GPU Scan.* This artifact contains testing material for three APIs :

- Google Chrome Dawn
- WGPU
- CUDA

---

# Building the Dawn Artifact
You have two options to build the Dawn artifact:
- Manual Build: Follow the detailed steps below to clone the Dawn repository, build and install Dawn, and then build the artifact.
- Automated Build Script: Use the provided build script located in this directory. Simply run the script with the number of parallel build threads (default is 4).

## Device Requirements
- Any device supporting Dawn, with subgroup and timestamp query capabilities.
- At least **384 MB** of available device memory.
- At least **2 GB** of disk space is required to build Dawn.

## Software Requirements
- Git
- CMake 3.10.2+ (or another C++ build tool)
- A C++17-compliant compiler (Clang recommended)
- Python for fetching dependencies

## Building Dawn
Although Dawn’s native WebGPU is supported on Windows, macOS, and Linux, this section will focus on building it in Unix-like environments for brevity, using CMake as the example build tool and Clang++ as the example compiler.

### 0. Install dependencies (Ubuntu package names)

```sh
sudo apt install libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libx11-xcb-dev mesa-common-dev
```

### 1. Clone the Dawn repository

```sh
git clone https://dawn.googlesource.com/dawn
cd dawn
git fetch --all
git checkout chromium/6950
```

### 2. Build and install Dawn

```sh
cmake -S . -B out/Release \
-DDAWN_FETCH_DEPENDENCIES=ON \
-DDAWN_ENABLE_INSTALL=ON \
-DCMAKE_BUILD_TYPE=Release \
-DDAWN_BUILD_TESTS=OFF \
-DDAWN_BUILD_SAMPLES=OFF \
-DTINT_BUILD_TESTS=OFF \
-DTINT_BUILD_CMD_TOOLS=OFF \
-DCMAKE_CXX_COMPILER=clang++

cmake --build out/Release --parallel <num_threads>

cmake --install out/Release --prefix install/Release
```

### 3. Building the Artifact
After successfully building and installing Dawn, follow these steps to build the artifact:

```sh
cd Decoupled-Fallback-Paper/artifact/Dawn
export CMAKE_PREFIX_PATH=/path/to/dawn/install/Release

cmake -S . -B out/Release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++
cmake --build out/Release
```
    
## Running the Artifact
Once the artifact has been successfully built, you can run it using the following command:

```sh
./out/Release/dawn <TestType> [record] [deviceName]

# Arguments:
# <TestType>   Specifies the test to run. Valid options are:
#    csdl         - Runs the Decoupled Lookback test. 
#                   Warning: this will crash without FPG.
#    csdldf       - Runs the Decoupled Fallback test.
#    full         - Runs Decoupled Fallback and simulated blocking tests.
#    sizecsdldf   - Runs the Chained-Scan Fallback test over a range of sizes 
#                   (from 2^10 to 2^25), recording performance for each size.
#    sizememcpy   - Runs a Memcpy test over a range of sizes (from 2^10 to 2^25),
#                   recording performance for each size.
#
# [record]        (Optional) Enables recording of test results to a CSV file.
# [deviceName]    (Optional) Appends the device name to recorded test results.
```

# Building the WGPU Artifact

## Device Requirements
- Any device supporting WGPU, with subgroup and timestamp query capabilities.
- At least **384 MB** of available device memory.
- At least **300 MB** of disk space is required to build WGPU.

## Software Requirements
- Rust

## Building/Running the Artifact

```sh
cd Decoupled-Fallback-Paper/artifact/WGPU

cargo run --release <TestType> [record] [deviceName]

# Arguments:
# <TestType>   Specifies the test to run. Valid options are:
#    csdl      - Runs the Decoupled Lookback test.
#                Warning: this will crash without FPG.
#    csdldf    - Runs the Decoupled Fallback test.
#    full      - Runs Decoupled Fallback and simulated blocking tests.
#
# [record]     (Optional) Enables recording of test results to a CSV file.
# [deviceName] (Optional) Appends the device name to recorded test results.
```

# Building the CUDA Artifact

## Device Requirements
- Any CUDA-capable device with Compute Capability 7.x or higher
- At least **384 MB** of available device memory

## Software Requirements
- CUDA Toolkit
- Make (or other build tools)

## Building the Artifact

```sh
cd Decoupled-Fallback-Paper/artifact/CUDA
make
```

## Running the Artifact
```sh
./main.out <testType> [record [csvName]]

# Arguments:
# <testType>   Specifies the test to run. Valid options are:
#    cub       - Runs the CUB scan test.
#    csdldf    - Runs the Decoupled Fallback scan test.
#    memcpy    - Runs the CUDA API Memcpy performance test.
#
# [record]     (Optional) Enables recording of test results to a CSV file.
# [csvName]    (Optional) Appends the device name or custom prefix to recorded test results.
```
