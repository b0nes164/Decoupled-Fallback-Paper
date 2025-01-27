#include <dawn/webgpu_cpp.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

constexpr uint32_t MAX_EMULATE = 10;

struct GPUContext {
    wgpu::Instance instance;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::QuerySet querySet;
};

struct ComputeShader {
    wgpu::BindGroup bindGroup;
    wgpu::ComputePipeline computePipeline;
    std::string label;
};

struct Shaders {
    ComputeShader init;
    ComputeShader reduce;
    ComputeShader spineScan;
    ComputeShader downsweep;
    ComputeShader csdl;
    ComputeShader csdldf;
    ComputeShader csdldfOcc;
    ComputeShader csdldfEmulate[MAX_EMULATE + 1];
    ComputeShader memcpy;
    ComputeShader validate;
};

struct GPUBuffers {
    wgpu::Buffer info;
    wgpu::Buffer scanIn;
    wgpu::Buffer scanOut;
    wgpu::Buffer scanBump;
    wgpu::Buffer reduction;
    wgpu::Buffer timestamp;
    wgpu::Buffer readbackTimestamp;
    wgpu::Buffer readback;
    wgpu::Buffer misc;
};

struct TestArgs {
    GPUContext& gpu;
    GPUBuffers& buffs;
    Shaders& shaders;
    uint32_t size;
    uint32_t batchSize;
    uint32_t threadBlocks;
    uint32_t readbackSize;
    uint32_t fallbackPreset;
    bool shouldValidate = false;
    bool shouldReadback = false;
    bool shouldTime = false;
    bool shouldGetStats = false;
    bool shouldRecord = false;
};

struct DataStruct {
    std::vector<double> time;
    std::vector<uint32_t> totalSpins;
    std::vector<double> lookbackLength;
    std::vector<uint32_t> fallbacksInitiated;
    std::vector<uint32_t> successfulInsertions;
};

void GetGPUContext(GPUContext* context, uint32_t timestampCount) {
    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.features.timedWaitAnyEnable = true;
    wgpu::Instance instance = wgpu::CreateInstance(&instanceDescriptor);
    if (instance == nullptr) {
        std::cerr << "Instance creation failed!\n";
    }

    wgpu::RequestAdapterOptions options = {};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    options.backendType = wgpu::BackendType::Undefined;  // specify as needed

    wgpu::Adapter adapter;
    std::promise<void> adaptPromise;
    instance.RequestAdapter(
        &options, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapt,
            wgpu::StringView) {
            if (status == wgpu::RequestAdapterStatus::Success) {
                adapter = adapt;
            } else {
                std::cerr << "Failed to get adapter" << std::endl;
            }
            adaptPromise.set_value();
        });
    std::future<void> adaptFuture = adaptPromise.get_future();
    while (adaptFuture.wait_for(std::chrono::microseconds(100)) ==
           std::future_status::timeout) {
        instance.ProcessEvents();
    }

    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    std::cout << "VendorID: " << std::hex << info.vendorID << std::dec
              << std::endl;
    std::cout << "Vendor: " << std::string(info.vendor.data, info.vendor.length)
              << std::endl;
    std::cout << "Architecture: "
              << std::string(info.architecture.data, info.architecture.length)
              << std::endl;
    std::cout << "DeviceID: " << std::hex << info.deviceID << std::dec
              << std::endl;
    std::cout << "Name: " << std::string(info.device.data, info.device.length)
              << std::endl;
    std::cout << "Driver description: "
              << std::string(info.description.data, info.description.length)
              << std::endl;
    std::cout << "Backend "
              << (info.backendType == wgpu::BackendType::Vulkan ? "vk"
                                                                : "not vk")
              << std::endl;  // LOL

    std::vector<wgpu::FeatureName> reqFeatures = {
        wgpu::FeatureName::Subgroups,
        wgpu::FeatureName::TimestampQuery,
    };

    auto errorCallback = [](const wgpu::Device& device, wgpu::ErrorType type,
                            wgpu::StringView message) {
        std::cerr << "Error: " << std::string(message.data, message.length)
                  << std::endl;
    };

    wgpu::DeviceDescriptor devDescriptor{};
    devDescriptor.requiredFeatures = reqFeatures.data();
    devDescriptor.requiredFeatureCount =
        static_cast<uint32_t>(reqFeatures.size());
    devDescriptor.SetUncapturedErrorCallback(errorCallback);

    wgpu::Device device;
    std::promise<void> devPromise;
    adapter.RequestDevice(
        &devDescriptor, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device dev,
            wgpu::StringView) {
            if (status == wgpu::RequestDeviceStatus::Success) {
                device = dev;
            } else {
                std::cerr << "Failed to get device" << std::endl;
            }
            devPromise.set_value();
        });
    std::future<void> devFuture = devPromise.get_future();
    while (devFuture.wait_for(std::chrono::microseconds(100)) ==
           std::future_status::timeout) {
        instance.ProcessEvents();
    }
    wgpu::Queue queue = device.GetQueue();

    // Query set for the timing, no need to grab the timing period/frequency?
    wgpu::QuerySetDescriptor querySetDescriptor{};
    querySetDescriptor.label = "Timestamp Query Set";
    querySetDescriptor.count = timestampCount * 2;
    querySetDescriptor.type = wgpu::QueryType::Timestamp;
    wgpu::QuerySet querySet = device.CreateQuerySet(&querySetDescriptor);

    (*context).instance = instance;
    (*context).device = device;
    (*context).queue = queue;
    (*context).querySet = querySet;
}

void GetGPUBuffers(const wgpu::Device& device, GPUBuffers* buffs,
                   uint32_t threadBlocks, uint32_t timestampCount,
                   uint32_t size, uint32_t miscSize, uint32_t maxReadbackSize) {
    wgpu::BufferDescriptor infoDesc = {};
    infoDesc.label = "Info";
    infoDesc.size = sizeof(uint32_t) * 3;
    infoDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer info = device.CreateBuffer(&infoDesc);

    wgpu::BufferDescriptor scanInDesc = {};
    scanInDesc.label = "Scan Input";
    scanInDesc.size = sizeof(uint32_t) * size;
    scanInDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer scanIn = device.CreateBuffer(&scanInDesc);

    wgpu::BufferDescriptor scanOutDesc = {};
    scanOutDesc.label = "Scan Output";
    scanOutDesc.size = sizeof(uint32_t) * size;
    scanOutDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer scanOut = device.CreateBuffer(&scanOutDesc);

    wgpu::BufferDescriptor scanBumpDesc = {};
    scanBumpDesc.label = "Scan Atomic Bump";
    scanBumpDesc.size = sizeof(uint32_t);
    scanBumpDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer scanBump = device.CreateBuffer(&scanBumpDesc);

    wgpu::BufferDescriptor redDesc = {};
    redDesc.label = "Intermediate Reduction";
    redDesc.size = sizeof(uint32_t) * threadBlocks *
                   4;  // To accomodate struct version of CSDLDF, allocate
    redDesc.usage =
        wgpu::BufferUsage::Storage;  // more memory than is necessary for others
    wgpu::Buffer reduction = device.CreateBuffer(&redDesc);

    wgpu::BufferDescriptor timestampDesc = {};
    timestampDesc.label = "Timestamp";
    timestampDesc.size = sizeof(uint64_t) * timestampCount * 2;
    timestampDesc.usage =
        wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer timestamp = device.CreateBuffer(&timestampDesc);

    wgpu::BufferDescriptor timestampReadDesc = {};
    timestampReadDesc.label = "Timestamp Readback";
    timestampReadDesc.size = sizeof(uint64_t) * timestampCount * 2;
    timestampReadDesc.usage =
        wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer timestampReadback = device.CreateBuffer(&timestampReadDesc);

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.label = "Main Readback";
    readbackDesc.size = sizeof(uint32_t) * maxReadbackSize;
    readbackDesc.usage =
        wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer readback = device.CreateBuffer(&readbackDesc);

    wgpu::BufferDescriptor miscDesc = {};
    miscDesc.label = "Miscellaneous";
    miscDesc.size = sizeof(uint32_t) * miscSize;
    miscDesc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer misc = device.CreateBuffer(&miscDesc);

    (*buffs).info = info;
    (*buffs).scanIn = scanIn;
    (*buffs).scanOut = scanOut;
    (*buffs).scanBump = scanBump;
    (*buffs).reduction = reduction;
    (*buffs).timestamp = timestamp;
    (*buffs).readbackTimestamp = timestampReadback;
    (*buffs).readback = readback;
    (*buffs).misc = misc;
}

// For simplicity we will use the same brind group and layout for all kernels
void GetComputeShaderPipeline(const wgpu::Device& device,
                              const GPUBuffers& buffs, ComputeShader* cs,
                              const char* entryPoint,
                              const wgpu::ShaderModule& module,
                              const std::string& csLabel) {
    auto makeLabel = [&](const std::string& suffix) -> std::string {
        return csLabel + suffix;
    };

    wgpu::BindGroupLayoutEntry bglInfo = {};
    bglInfo.binding = 0;
    bglInfo.visibility = wgpu::ShaderStage::Compute;
    bglInfo.buffer.type = wgpu::BufferBindingType::Uniform;

    wgpu::BindGroupLayoutEntry bglScanIn = {};
    bglScanIn.binding = 1;
    bglScanIn.visibility = wgpu::ShaderStage::Compute;
    bglScanIn.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglScanOut = {};
    bglScanOut.binding = 2;
    bglScanOut.visibility = wgpu::ShaderStage::Compute;
    bglScanOut.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglScanBump = {};
    bglScanBump.binding = 3;
    bglScanBump.visibility = wgpu::ShaderStage::Compute;
    bglScanBump.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglReduction = {};
    bglReduction.binding = 4;
    bglReduction.visibility = wgpu::ShaderStage::Compute;
    bglReduction.buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutEntry bglMisc = {};
    bglMisc.binding = 5;
    bglMisc.visibility = wgpu::ShaderStage::Compute;
    bglMisc.buffer.type = wgpu::BufferBindingType::Storage;

    std::vector<wgpu::BindGroupLayoutEntry> bglEntries{
        bglInfo, bglScanIn, bglScanOut, bglScanBump, bglReduction, bglMisc};

    wgpu::BindGroupLayoutDescriptor bglDesc = {};
    bglDesc.label = makeLabel("Bind Group Layout").c_str();
    bglDesc.entries = bglEntries.data();
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    wgpu::BindGroupLayout bgl = device.CreateBindGroupLayout(&bglDesc);

    wgpu::BindGroupEntry bgInfo = {};
    bgInfo.binding = 0;
    bgInfo.buffer = buffs.info;
    bgInfo.size = buffs.info.GetSize();

    wgpu::BindGroupEntry bgScanIn = {};
    bgScanIn.binding = 1;
    bgScanIn.buffer = buffs.scanIn;
    bgScanIn.size = buffs.scanIn.GetSize();

    wgpu::BindGroupEntry bgScanOut = {};
    bgScanOut.binding = 2;
    bgScanOut.buffer = buffs.scanOut;
    bgScanOut.size = buffs.scanOut.GetSize();

    wgpu::BindGroupEntry bgScanBump = {};
    bgScanBump.binding = 3;
    bgScanBump.buffer = buffs.scanBump;
    bgScanBump.size = buffs.scanBump.GetSize();

    wgpu::BindGroupEntry bgReduction = {};
    bgReduction.binding = 4;
    bgReduction.buffer = buffs.reduction;
    bgReduction.size = buffs.reduction.GetSize();

    wgpu::BindGroupEntry bgMisc = {};
    bgMisc.binding = 5;
    bgMisc.buffer = buffs.misc;
    bgMisc.size = buffs.misc.GetSize();

    std::vector<wgpu::BindGroupEntry> bgEntries{
        bgInfo, bgScanIn, bgScanOut, bgScanBump, bgReduction, bgMisc};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.entries = bgEntries.data();
    bindGroupDesc.entryCount = static_cast<uint32_t>(bgEntries.size());
    bindGroupDesc.layout = bgl;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

    wgpu::PipelineLayoutDescriptor pipeLayoutDesc = {};
    pipeLayoutDesc.label = makeLabel("Pipeline Layout").c_str();
    pipeLayoutDesc.bindGroupLayoutCount = 1;
    pipeLayoutDesc.bindGroupLayouts = &bgl;
    wgpu::PipelineLayout pipeLayout =
        device.CreatePipelineLayout(&pipeLayoutDesc);

    wgpu::ComputeState compState = {};
    compState.entryPoint = entryPoint;
    compState.module = module;

    wgpu::ComputePipelineDescriptor compPipeDesc = {};
    compPipeDesc.label = makeLabel("Compute Pipeline").c_str();
    compPipeDesc.layout = pipeLayout;
    compPipeDesc.compute = compState;
    wgpu::ComputePipeline compPipeline =
        device.CreateComputePipeline(&compPipeDesc);

    (*cs).bindGroup = bindGroup;
    (*cs).computePipeline = compPipeline;
    (*cs).label = csLabel;
}

std::string ReadWGSL(const std::string& path,
                     const std::vector<std::string>& pseudoArgs) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return "";
    }

    std::stringstream buffer;
    buffer << "enable subgroups;\n";  // Enable subgroups here.
    for (size_t i = 0; i < pseudoArgs.size(); ++i) {
        buffer << pseudoArgs[i] << "\n";
    }
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

void CreateShaderFromSource(const GPUContext& gpu, const GPUBuffers& buffs,
                            ComputeShader* cs, const char* entryPoint,
                            const std::string& path, const std::string& csLabel,
                            const std::vector<std::string>& pseudoArgs) {
    wgpu::ShaderSourceWGSL wgslSource = {};
    std::string source = ReadWGSL(path, pseudoArgs);
    wgslSource.code = source.c_str();
    wgpu::ShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgslSource;
    wgpu::ShaderModule mod = gpu.device.CreateShaderModule(&desc);
    std::promise<void> promise;
    mod.GetCompilationInfo(
        wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::CompilationInfoRequestStatus status,
            wgpu::CompilationInfo const* info) {
            for (size_t i = 0; i < info->messageCount; ++i) {
                const wgpu::CompilationMessage& message = info->messages[i];
                if (message.type == wgpu::CompilationMessageType::Error) {
                    std::cerr << "Shader compilation error: "
                              << std::string(message.message.data,
                                             message.message.length)
                              << std::endl;
                } else if (message.type ==
                           wgpu::CompilationMessageType::Warning) {
                    std::cerr << "Shader compilation warning: "
                              << std::string(message.message.data,
                                             message.message.length)
                              << std::endl;
                }
            }
            promise.set_value();
        });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::microseconds(100)) ==
           std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
    GetComputeShaderPipeline(gpu.device, buffs, cs, entryPoint, mod, csLabel);
}

void GetAllShaders(const GPUContext& gpu, const GPUBuffers& buffs,
                   Shaders* shaders) {
    std::vector<std::string> empty;
    CreateShaderFromSource(gpu, buffs, &shaders->init, "main",
                           "Shaders/init.wgsl", "Init", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->reduce, "reduce",
                           "Shaders/rts.wgsl", "Reduce", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->spineScan, "spine_scan",
                           "Shaders/rts.wgsl", "Spine Scan", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->downsweep, "downsweep",
                           "Shaders/rts.wgsl", "Downsweep", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdl, "main",
                           "Shaders/csdl.wgsl", "CSDL", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdldf, "main",
                           "Shaders/csdldf.wgsl", "CSDLDF", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdldfOcc, "main",
                           "Shaders/csdldf_occ.wgsl", "CSDLDF OCC", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->memcpy, "main",
                           "Shaders/memcpy.wgsl", "Memcpy", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->validate, "main",
                           "Shaders/validate.wgsl", "Validate", empty);

    // Create shaders with decreasing artificial deadlocking
    for (uint32_t i = 0; i < MAX_EMULATE; ++i) {
        std::vector<std::string> t;
        t.push_back("const DEADLOCK_MASK = " + std::to_string((1 << i) - 1) +
                    "u;");
        CreateShaderFromSource(gpu, buffs, &shaders->csdldfEmulate[i], "main",
                               "Shaders/csdldf_emulate.wgsl",
                               "CSDLDF Emulation", t);
    }

    // hardcode the last shader with 0xffffffff for NO deadlocking
    std::vector<std::string> t;
    t.push_back("const DEADLOCK_MASK = 0xffffffffu;");
    CreateShaderFromSource(gpu, buffs, &shaders->csdldfEmulate[MAX_EMULATE],
                           "main", "Shaders/csdldf_emulate.wgsl",
                           "CSDLDF Emulation", t);
}

void SetComputePass(const ComputeShader& cs, wgpu::CommandEncoder* comEncoder,
                    uint32_t threadBlocks) {
    wgpu::ComputePassDescriptor comDesc = {};
    comDesc.label = cs.label.c_str();
    wgpu::ComputePassEncoder pass = (*comEncoder).BeginComputePass(&comDesc);
    pass.SetPipeline(cs.computePipeline);
    pass.SetBindGroup(0, cs.bindGroup);
    pass.DispatchWorkgroups(threadBlocks, 1, 1);
    pass.End();
}

void SetComputePassTimed(const ComputeShader& cs,
                         wgpu::CommandEncoder* comEncoder,
                         const wgpu::QuerySet& querySet, uint32_t threadBlocks,
                         uint32_t timeStampOffset) {
    wgpu::ComputePassTimestampWrites timeStamp = {};
    timeStamp.beginningOfPassWriteIndex = timeStampOffset * 2;
    timeStamp.endOfPassWriteIndex = timeStampOffset * 2 + 1;
    timeStamp.querySet = querySet;
    wgpu::ComputePassDescriptor comDesc = {};
    comDesc.label = cs.label.c_str();
    comDesc.timestampWrites = &timeStamp;
    wgpu::ComputePassEncoder pass = (*comEncoder).BeginComputePass(&comDesc);
    pass.SetPipeline(cs.computePipeline);
    pass.SetBindGroup(0, cs.bindGroup);
    pass.DispatchWorkgroups(threadBlocks, 1, 1);
    pass.End();
}

void QueueSync(const GPUContext& gpu) {
    std::promise<void> promise;
    gpu.queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::QueueWorkDoneStatus status) {
            if (status != wgpu::QueueWorkDoneStatus::Success) {
                std::cerr << "uh oh" << std::endl;
            }
            promise.set_value();
        });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::microseconds(100)) ==
           std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

void CopyBufferSync(const GPUContext& gpu, wgpu::Buffer* srcReadback,
                    wgpu::Buffer* dstReadback, uint64_t sourceOffsetBytes,
                    uint64_t copySizeBytes) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Copy Command Encoder";
    wgpu::CommandEncoder comEncoder =
        gpu.device.CreateCommandEncoder(&comEncDesc);
    comEncoder.CopyBufferToBuffer(*srcReadback, sourceOffsetBytes, *dstReadback,
                                  0ULL, copySizeBytes);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(1, &comBuffer);
    QueueSync(gpu);
}

template <typename T>
void ReadbackSync(const GPUContext& gpu, wgpu::Buffer* dstReadback,
                  std::vector<T>* readOut, uint64_t readbackSizeBytes) {
    std::promise<void> promise;
    dstReadback->MapAsync(
        wgpu::MapMode::Read, 0, readbackSizeBytes,
        wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
                const void* data =
                    dstReadback->GetConstMappedRange(0, readbackSizeBytes);
                std::memcpy(readOut->data(), data, readbackSizeBytes);
                dstReadback->Unmap();
            } else {
                std::cerr << "Bad readback" << std::endl;
            }
            promise.set_value();
        });

    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::microseconds(100)) ==
           std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

template <typename T>
void CopyAndReadbackSync(const GPUContext& gpu, wgpu::Buffer* srcReadback,
                         wgpu::Buffer* dstReadback, std::vector<T>* readOut,
                         uint32_t sourceOffset, uint32_t readbackSize) {
    CopyBufferSync(gpu, srcReadback, dstReadback, sourceOffset * sizeof(T),
                   readbackSize * sizeof(T));
    ReadbackSync(gpu, dstReadback, readOut, readbackSize * sizeof(T));
}

bool Validate(const GPUContext& gpu, GPUBuffers* buffs,
              const ComputeShader& validate) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Validate Command Encoder";
    wgpu::CommandEncoder comEncoder =
        gpu.device.CreateCommandEncoder(&comEncDesc);
    SetComputePass(validate, &comEncoder, 256);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(1, &comBuffer);
    QueueSync(gpu);

    std::vector<uint32_t> readOut(1, 1);
    CopyAndReadbackSync(gpu, &buffs->misc, &buffs->readback, &readOut, 0, 1);
    bool testPassed = readOut[0] == 0;
    if (!testPassed) {
        std::cerr << "Test failed: " << readOut[0] << " errors" << std::endl;
    }
    return testPassed;
}

void ReadbackAndPrintSync(const GPUContext& gpu, GPUBuffers* buffs,
                          uint32_t readbackSize) {
    std::vector<uint32_t> readOut(readbackSize);
    CopyAndReadbackSync(gpu, &buffs->scanOut, &buffs->readback, &readOut, 0,
                        readbackSize);
    for (uint32_t i = 0; i < (readbackSize + 31) / 32; ++i) {
        for (uint32_t k = 0; k < 32; ++k) {
            std::cout << readOut[i * 32 + k] << ", ";
        }
        std::cout << std::endl;
    }
}

void ResolveTimestampQuery(GPUBuffers* buffs, const wgpu::QuerySet& query,
                           const wgpu::CommandEncoder* comEncoder,
                           uint32_t passCount) {
    uint32_t entriesToResolve = passCount * 2;
    (*comEncoder)
        .ResolveQuerySet(query, 0, entriesToResolve, buffs->timestamp, 0ULL);
    (*comEncoder)
        .CopyBufferToBuffer(buffs->timestamp, 0ULL, buffs->readbackTimestamp,
                            0ULL, entriesToResolve * sizeof(uint64_t));
}

uint64_t GetTime(const GPUContext& gpu, GPUBuffers* buffs, uint32_t passCount) {
    uint64_t totalTime = 0ULL;
    std::vector<uint64_t> timeOut(passCount * 2);
    ReadbackSync(gpu, &buffs->readbackTimestamp, &timeOut,
                 passCount * 2 * sizeof(uint64_t));
    for (uint32_t i = 0; i < passCount; ++i) {
        totalTime += timeOut[i * 2 + 1] - timeOut[i * 2];
    }
    return totalTime;
}

void GetFallbackStatistics(const GPUContext& gpu, GPUBuffers* buffs,
                           std::vector<uint32_t>* stats) {
    CopyAndReadbackSync(gpu, &buffs->misc, &buffs->readback, stats, 1, 4);
}

void InitializeUniforms(const GPUContext& gpu, GPUBuffers* buffs, uint32_t size,
                        uint32_t threadBlocks) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Initialize Uniforms Command Encoder";
    wgpu::CommandEncoder comEncoder =
        gpu.device.CreateCommandEncoder(&comEncDesc);
    std::vector<uint32_t> info{size, (size + 3) / 4, threadBlocks};
    gpu.queue.WriteBuffer(buffs->info, 0ULL, info.data(),
                          info.size() * sizeof(uint32_t));
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(0, &comBuffer);
    QueueSync(gpu);
}

uint32_t RTS(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 3;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.reduce, comEncoder, args.gpu.querySet,
                            args.threadBlocks, 0);
        SetComputePassTimed(args.shaders.spineScan, comEncoder,
                            args.gpu.querySet, 1, 1);
        SetComputePassTimed(args.shaders.downsweep, comEncoder,
                            args.gpu.querySet, args.threadBlocks, 2);
    } else {
        SetComputePass(args.shaders.reduce, comEncoder, args.threadBlocks);
        SetComputePass(args.shaders.spineScan, comEncoder, 1);
        SetComputePass(args.shaders.downsweep, comEncoder, args.threadBlocks);
    }
    return passCount;
}

uint32_t CSDL(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdl, comEncoder, args.gpu.querySet,
                            args.threadBlocks, 0);
    } else {
        SetComputePass(args.shaders.csdl, comEncoder, args.threadBlocks);
    }
    return passCount;
}

uint32_t CSDLDF(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldf, comEncoder, args.gpu.querySet,
                            args.threadBlocks, 0);
    } else {
        SetComputePass(args.shaders.csdldf, comEncoder, args.threadBlocks);
    }
    return passCount;
}

uint32_t CSDLDFStats(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldfEmulate[MAX_EMULATE], comEncoder,
                            args.gpu.querySet, args.threadBlocks, 0);
    } else {
        SetComputePass(args.shaders.csdldfEmulate[MAX_EMULATE], comEncoder,
                       args.threadBlocks);
    }
    return passCount;
}

uint32_t CSDLDFEmulate(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldfEmulate[args.fallbackPreset],
                            comEncoder, args.gpu.querySet, args.threadBlocks,
                            0);
    } else {
        SetComputePass(args.shaders.csdldfEmulate[args.fallbackPreset],
                       comEncoder, args.threadBlocks);
    }
    return passCount;
}

uint32_t Memcpy(const TestArgs& args, wgpu::CommandEncoder* comEncoder) {
    const uint32_t passCount = 1;
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.memcpy, comEncoder, args.gpu.querySet,
                            args.threadBlocks, 0);
    } else {
        SetComputePass(args.shaders.memcpy, comEncoder, args.threadBlocks);
    }
    return passCount;
}

uint32_t GetOccupancySync(const TestArgs& args) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Command Encoder";
    wgpu::CommandEncoder comEncoder =
        args.gpu.device.CreateCommandEncoder(&comEncDesc);
    SetComputePass(args.shaders.init, &comEncoder, 256);
    SetComputePass(args.shaders.csdldfOcc, &comEncoder, args.threadBlocks);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    args.gpu.queue.Submit(1, &comBuffer);
    QueueSync(args.gpu);
    std::vector<uint32_t> readOut(1);
    CopyAndReadbackSync(args.gpu, &args.buffs.misc, &args.buffs.readback,
                        &readOut, 1, 1);
    std::cout << std::endl;
    std::cout << "Estimated CSDLDF Occupancy: " << readOut[0] << std::endl;
    return readOut[0];
}

void RecordToCSV(const TestArgs& args, const DataStruct& data,
                 const std::string& filename) {
    std::ofstream file(filename + ".csv");

    if (args.shouldGetStats) {
        // Write full headers
        file << "time,totalSpins,lookbackLength,fallbacksInitiated,"
                "successfulInsertions\n";

        // Write full data
        size_t rows = data.time.size();
        for (size_t i = 0; i < rows; ++i) {
            file << data.time[i] << "," << data.totalSpins[i] << ","
                 << data.lookbackLength[i] << "," << data.fallbacksInitiated[i]
                 << "," << data.successfulInsertions[i] << "\n";
        }
    } else {
        // Write minimal headers
        file << "time\n";

        // Write only time data
        for (double t : data.time) {
            file << t << "\n";
        }
    }

    file.close();
}

void Run(std::string testLabel, const TestArgs& args,
         uint32_t (*MainPass)(const TestArgs&, wgpu::CommandEncoder*)) {
    DataStruct data;
    if (args.shouldRecord) {
        data.time.resize(args.batchSize);
        data.totalSpins.resize(args.batchSize);
        data.lookbackLength.resize(args.batchSize);
        data.fallbacksInitiated.resize(args.batchSize);
        data.successfulInsertions.resize(args.batchSize);
    }

    uint32_t testsPassed = 0;
    uint64_t totalTime = 0ULL;
    uint32_t totalSpins = 0;
    uint32_t lookbackLength = 0;
    uint32_t fallbacksInitiated = 0;
    uint32_t successfulInsertions = 0;
    for (uint32_t i = 0; i <= args.batchSize; ++i) {
        wgpu::CommandEncoderDescriptor comEncDesc = {};
        comEncDesc.label = "Command Encoder";
        wgpu::CommandEncoder comEncoder =
            args.gpu.device.CreateCommandEncoder(&comEncDesc);
        SetComputePass(args.shaders.init, &comEncoder, 256);
        uint32_t passCount = MainPass(args, &comEncoder);
        if (args.shouldTime) {
            ResolveTimestampQuery(&args.buffs, args.gpu.querySet, &comEncoder,
                                  passCount);
        }
        wgpu::CommandBuffer comBuffer = comEncoder.Finish();
        args.gpu.queue.Submit(1, &comBuffer);
        QueueSync(args.gpu);

        // The first test is always discarded to prep caches and TLB
        if (i != 0) {
            if (args.shouldTime) {
                const uint64_t t = GetTime(args.gpu, &args.buffs, passCount);
                totalTime += t;
                if (args.shouldRecord) {
                    data.time[i - 1] = static_cast<double>(args.size) / t;
                }
            }

            if (args.shouldGetStats) {
                std::vector<uint32_t> stats(4, 0);
                GetFallbackStatistics(args.gpu, &args.buffs, &stats);
                totalSpins += stats[0];
                fallbacksInitiated += stats[1];
                successfulInsertions += stats[2];
                lookbackLength += stats[3];
                if (args.shouldRecord) {
                    data.totalSpins[i - 1] = stats[0];
                    data.fallbacksInitiated[i - 1] = stats[1];
                    data.successfulInsertions[i - 1] = stats[2];
                    data.lookbackLength[i - 1] =
                        static_cast<double>(stats[3]) / args.threadBlocks;
                }
            }

            if (args.shouldValidate) {
                testsPassed +=
                    Validate(args.gpu, &args.buffs, args.shaders.validate);
            }
        }
    }
    std::cout << std::endl;

    std::cout << testLabel << " Tests Complete at size: " << args.size << "\n";

    if (args.shouldReadback) {
        ReadbackAndPrintSync(args.gpu, &args.buffs, args.readbackSize);
    }

    if (args.shouldGetStats) {
        double avgTotalSpins = static_cast<double>(totalSpins) / args.batchSize;
        double avgLookbackLength = static_cast<double>(lookbackLength) /
                                   args.batchSize / args.threadBlocks;
        double avgFallbacksAttempted =
            static_cast<double>(fallbacksInitiated) / args.batchSize;
        double avgSuccessfulInsertions =
            static_cast<double>(successfulInsertions) / args.batchSize;
        std::cout << "Threadblocks Launched: " << args.threadBlocks
                  << std::endl;
        std::cout << "Average Total Spins: " << avgTotalSpins << std::endl;
        std::cout << "Average Lookback Length Per Workgroup: "
                  << avgLookbackLength << std::endl;
        std::cout << "Average Total Fallbacks Initiated: "
                  << avgFallbacksAttempted << std::endl;
        std::cout << "Average Total Successful Insertions: "
                  << avgSuccessfulInsertions << std::endl;
    }

    if (args.shouldValidate) {
        std::cout << testsPassed << "/" << args.batchSize;
        if (testsPassed == args.batchSize) {
            std::cout << " ALL TESTS PASSED" << std::endl;
        } else {
            std::cout << " TEST FAILED" << std::endl;
        }
    }

    if (args.shouldTime) {
        double dTime = static_cast<double>(totalTime);
        dTime /= 1e9;
        std::cout << "Total time elapsed " << dTime << std::endl;
        double speed =
            ((uint64_t)args.size * (uint64_t)(args.batchSize)) / dTime;
        printf("Estimated speed %e ele/s\n", speed);
    }

    if (args.shouldRecord) {
        RecordToCSV(args, data, testLabel);
    }
}

enum TestType {
    Csdl,
    Csdldf,
    Full,
};

auto printUsage = []() {
    std::cerr << "Usage: <TestType: \"csdl\"|\"csdldf\"|\"full\"> "
                 "[\"record\"] [deviceName]"
              << std::endl;
};

int main(int argc, char* argv[]) {
    constexpr uint32_t MISC_SIZE =
        5;  // Max scratch memory we use to track various stats
    constexpr uint32_t PART_SIZE =
        4096;  // MUST match the partition size specified in shaders.
    constexpr uint32_t MAX_TIMESTAMPS =
        3;  // Max number of passes to track with our query set
    constexpr uint32_t MAX_READBACK_SIZE =
        8192;  // Max size of our readback buffer

    // We allow:
    //   1) <TestType>                      e.g. "csdl"
    //   2) <TestType> record               e.g. "csdl record"
    //   3) <TestType> record <deviceName>  e.g. "csdl record myDevice"

    if (argc < 2 || argc > 4) {
        printUsage();
        return EXIT_FAILURE;
    }

    // Parse the test type
    TestType testType;
    std::string testTypeStr = argv[1];
    if (testTypeStr == "csdl") {
        testType = Csdl;
    } else if (testTypeStr == "csdldf") {
        testType = Csdldf;
    } else if (testTypeStr == "full") {
        testType = Full;
    } else {
        printUsage();
        return EXIT_FAILURE;
    }

    // Should Record?
    // Should Append Device Name?
    bool shouldRecord = false;
    std::string deviceName;
    if (argc >= 3) {
        if (std::string(argv[2]) == "record") {
            shouldRecord = true;
            if (argc == 4) {
                deviceName = std::string(argv[3]) + "_";
            }
        } else {
            printUsage();
            return EXIT_FAILURE;
        }
    }

    uint32_t size = 1 << 25;   // Size of the scan operation
    uint32_t batchSize = 500;  // How many tests to run
    uint32_t threadBlocks =    // Thread Blocks to launch based on input
        (size + PART_SIZE - 1) / PART_SIZE;
    uint32_t readbackSize =
        256;  // How many elements to readback, must be less than max

    // Test parameter controls
    bool shouldValidate = true;  // Perform validation?
    bool shouldReadback =
        false;               // Use readback to verify check results as needed
    bool shouldTime = true;  // Time results?

    GPUContext gpu;
    GPUBuffers buffs;
    Shaders shaders;
    GetGPUContext(&gpu, MAX_TIMESTAMPS);
    GetGPUBuffers(gpu.device, &buffs, threadBlocks, MAX_TIMESTAMPS, size,
                  MISC_SIZE, MAX_READBACK_SIZE);
    GetAllShaders(gpu, buffs, &shaders);
    InitializeUniforms(gpu, &buffs, size, threadBlocks);
    TestArgs args = {
        gpu,          buffs,        shaders,     size,           batchSize,
        threadBlocks, readbackSize, 0,           shouldValidate, shouldReadback,
        shouldTime,   false,        shouldRecord};

    // Run Memcopy kernel as a baseline.
    // We expect the speed of CSDL/CSDLDF
    // to approach or equal the speed of Memcopy
    args.shouldValidate = false;
    Run(deviceName + "Memcopy", args, Memcpy);
    args.shouldValidate = shouldValidate;

    switch (testType) {
        case Csdl:
            Run(deviceName + "CSDL", args, CSDL);
            break;
        case Csdldf:
            GetOccupancySync(args);
            Run(deviceName + "CSDLDF", args, CSDLDF);
            break;
        case Full:
            Run(deviceName + "RTS", args, RTS);

            GetOccupancySync(args);
            Run(deviceName + "CSDLDF", args, CSDLDF);

            args.shouldGetStats = true;
            Run(deviceName + "CSDLDF_Stats", args, CSDLDFStats);
            for (int32_t i = MAX_EMULATE - 1; i >= 0; --i) {
                args.fallbackPreset = static_cast<uint32_t>(i);
                Run(deviceName + "CSDLDF_" + std::to_string(1 << i), args,
                    CSDLDFEmulate);
            }
            break;
        default:
            break;
    }
    return EXIT_SUCCESS;
}
