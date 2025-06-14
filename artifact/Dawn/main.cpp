#include <dawn/webgpu_cpp.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

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
    ComputeShader initPass;
    ComputeShader initBatch;
    ComputeShader reduce;
    ComputeShader spineScan;
    ComputeShader downsweep;
    ComputeShader csdl;
    ComputeShader csdldf;
    ComputeShader csdldfOcc;
    ComputeShader csdldfSimulate;
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

enum TestType {
    Rts,
    Csdl,
    Csdldf,
    Full,
    SizeCsdldf,
    SizeMemcpy,
};

struct TestArgs {
    GPUContext& gpu;
    GPUBuffers& buffs;
    Shaders& shaders;
    uint32_t size;
    uint32_t workTiles;
    uint32_t readbackSize;
    uint32_t warmupRuns;
    uint32_t totalRuns;
    uint32_t batchSize;
    uint32_t maxQueryEntries;
    uint32_t miscStride;
    bool shouldValidate = false;
    bool shouldReadback = false;
    bool shouldTime = false;
    bool shouldGetStats = false;
    bool shouldRecord = false;
};

struct TestStatistics {
    uint64_t totalTime = 0ULL;
    uint64_t maxTime = 0ULL;
    uint64_t minTime = ~0ULL;
    std::map<uint64_t, unsigned int> timeMap;
    double totalSpins = 0.0;
    double totalLookbackLength = 0.0;
    uint64_t totalFallbacksInitiated = 0;
    uint64_t totalSuccessfulInsertions = 0;
};

struct DataStruct {
    std::vector<double> time;
    std::vector<double> totalSpins;
    std::vector<double> lookbackLength;
    std::vector<uint32_t> fallbacksInitiated;
    std::vector<uint32_t> successfulInsertions;

    DataStruct(const TestArgs& args) {
        if (args.shouldRecord) {
            time.resize(args.totalRuns);
            totalSpins.resize(args.totalRuns);
            lookbackLength.resize(args.totalRuns);
            fallbacksInitiated.resize(args.totalRuns);
            successfulInsertions.resize(args.totalRuns);
        }
    }
};

void GetGPUContext(GPUContext* context, uint32_t maxQueryEntries) {
    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.capabilities.timedWaitAnyEnable = true;
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
        [&](wgpu::RequestAdapterStatus status, wgpu::Adapter adapt, wgpu::StringView) {
            if (status == wgpu::RequestAdapterStatus::Success) {
                adapter = adapt;
            } else {
                std::cerr << "Failed to get adapter" << std::endl;
            }
            adaptPromise.set_value();
        });
    std::future<void> adaptFuture = adaptPromise.get_future();
    while (adaptFuture.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        instance.ProcessEvents();
    }

    wgpu::AdapterInfo info{};
    adapter.GetInfo(&info);
    std::cout << "VendorID: " << std::hex << info.vendorID << std::dec << std::endl;
    std::cout << "Vendor: " << std::string(info.vendor.data, info.vendor.length) << std::endl;
    std::cout << "Architecture: " << std::string(info.architecture.data, info.architecture.length)
              << std::endl;
    std::cout << "DeviceID: " << std::hex << info.deviceID << std::dec << std::endl;
    std::cout << "Name: " << std::string(info.device.data, info.device.length) << std::endl;
    std::cout << "Driver description: "
              << std::string(info.description.data, info.description.length) << std::endl;
    std::cout << "Backend " << (info.backendType == wgpu::BackendType::Vulkan ? "vk" : "not vk")
              << std::endl;  // LOL

    std::vector<wgpu::FeatureName> reqFeatures = {
        wgpu::FeatureName::Subgroups,
        wgpu::FeatureName::TimestampQuery,
    };

    auto errorCallback = [](const wgpu::Device& device, wgpu::ErrorType type,
                            wgpu::StringView message) {
        std::cerr << "Error: " << std::string(message.data, message.length) << std::endl;
    };

    wgpu::DawnTogglesDescriptor toggles = {};
    std::vector<const char*> enabled_toggles;
    enabled_toggles.push_back("allow_unsafe_apis");
    enabled_toggles.push_back("disable_robustness");
    enabled_toggles.push_back("disable_workgroup_init");
    enabled_toggles.push_back("skip_validation");
    toggles.enabledToggleCount = enabled_toggles.size();
    toggles.enabledToggles = enabled_toggles.data();

    std::vector<const char*> disabled_toggles;
    disabled_toggles.push_back("timestamp_quantization");
    toggles.disabledToggleCount = disabled_toggles.size();
    toggles.disabledToggles = disabled_toggles.data();

    wgpu::DeviceDescriptor devDescriptor{};
    devDescriptor.nextInChain = &toggles;
    devDescriptor.requiredFeatures = reqFeatures.data();
    devDescriptor.requiredFeatureCount = static_cast<uint32_t>(reqFeatures.size());
    devDescriptor.SetUncapturedErrorCallback(errorCallback);

    wgpu::Device device;
    std::promise<void> devPromise;
    adapter.RequestDevice(
        &devDescriptor, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::RequestDeviceStatus status, wgpu::Device dev, wgpu::StringView) {
            if (status == wgpu::RequestDeviceStatus::Success) {
                device = dev;
            } else {
                std::cerr << "Failed to get device" << std::endl;
            }
            devPromise.set_value();
        });
    std::future<void> devFuture = devPromise.get_future();
    while (devFuture.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        instance.ProcessEvents();
    }
    wgpu::Queue queue = device.GetQueue();

    wgpu::QuerySetDescriptor querySetDescriptor{};
    querySetDescriptor.label = "Timestamp Query Set";
    querySetDescriptor.count = maxQueryEntries;
    querySetDescriptor.type = wgpu::QueryType::Timestamp;
    wgpu::QuerySet querySet = device.CreateQuerySet(&querySetDescriptor);

    (*context).instance = instance;
    (*context).device = device;
    (*context).queue = queue;
    (*context).querySet = querySet;
}

void GetGPUBuffers(const wgpu::Device& device, GPUBuffers* buffs, uint32_t workTiles,
                   uint32_t maxQueryEntries, uint32_t size, uint32_t miscStride,
                   uint32_t maxReadbackSize) {
    wgpu::BufferDescriptor infoDesc = {};
    infoDesc.label = "Info";
    infoDesc.size = sizeof(uint32_t) * 4;
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
    scanBumpDesc.size = sizeof(uint32_t) * 2;
    scanBumpDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer scanBump = device.CreateBuffer(&scanBumpDesc);

    wgpu::BufferDescriptor redDesc = {};
    redDesc.label = "Intermediate Reduction";
    redDesc.size = sizeof(uint32_t) * workTiles * 2;
    redDesc.usage = wgpu::BufferUsage::Storage;
    wgpu::Buffer reduction = device.CreateBuffer(&redDesc);

    wgpu::BufferDescriptor timestampDesc = {};
    timestampDesc.label = "Timestamp";
    timestampDesc.size = sizeof(uint64_t) * maxQueryEntries * 2;
    timestampDesc.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
    wgpu::Buffer timestamp = device.CreateBuffer(&timestampDesc);

    wgpu::BufferDescriptor timestampReadDesc = {};
    timestampReadDesc.label = "Timestamp Readback";
    timestampReadDesc.size = sizeof(uint64_t) * maxQueryEntries * 2;
    timestampReadDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer timestampReadback = device.CreateBuffer(&timestampReadDesc);

    wgpu::BufferDescriptor readbackDesc = {};
    readbackDesc.label = "Main Readback";
    readbackDesc.size = sizeof(uint32_t) * maxReadbackSize;
    readbackDesc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
    wgpu::Buffer readback = device.CreateBuffer(&readbackDesc);

    wgpu::BufferDescriptor miscDesc = {};
    miscDesc.label = "Miscellaneous";
    miscDesc.size = sizeof(uint32_t) * miscStride * (maxQueryEntries / 2 + 1);
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
void GetComputeShaderPipeline(const wgpu::Device& device, const GPUBuffers& buffs,
                              ComputeShader* cs, const char* entryPoint,
                              const wgpu::ShaderModule& module, const std::string& csLabel) {
    auto makeLabel = [&](const std::string& suffix) -> std::string { return csLabel + suffix; };

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

    std::vector<wgpu::BindGroupLayoutEntry> bglEntries{bglInfo,     bglScanIn,    bglScanOut,
                                                       bglScanBump, bglReduction, bglMisc};

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

    std::vector<wgpu::BindGroupEntry> bgEntries{bgInfo,     bgScanIn,    bgScanOut,
                                                bgScanBump, bgReduction, bgMisc};

    wgpu::BindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.entries = bgEntries.data();
    bindGroupDesc.entryCount = static_cast<uint32_t>(bgEntries.size());
    bindGroupDesc.layout = bgl;
    wgpu::BindGroup bindGroup = device.CreateBindGroup(&bindGroupDesc);

    wgpu::PipelineLayoutDescriptor pipeLayoutDesc = {};
    pipeLayoutDesc.label = makeLabel("Pipeline Layout").c_str();
    pipeLayoutDesc.bindGroupLayoutCount = 1;
    pipeLayoutDesc.bindGroupLayouts = &bgl;
    wgpu::PipelineLayout pipeLayout = device.CreatePipelineLayout(&pipeLayoutDesc);

    wgpu::ComputeState compState = {};
    compState.entryPoint = entryPoint;
    compState.module = module;

    wgpu::ComputePipelineDescriptor compPipeDesc = {};
    compPipeDesc.label = makeLabel("Compute Pipeline").c_str();
    compPipeDesc.layout = pipeLayout;
    compPipeDesc.compute = compState;
    wgpu::ComputePipeline compPipeline = device.CreateComputePipeline(&compPipeDesc);

    (*cs).bindGroup = bindGroup;
    (*cs).computePipeline = compPipeline;
    (*cs).label = csLabel;
}

std::string ReadWGSL(const std::string& path, const std::vector<std::string>& pseudoArgs) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return "";
    }

    std::stringstream buffer;
    for (size_t i = 0; i < pseudoArgs.size(); ++i) {
        buffer << pseudoArgs[i] << "\n";
    }
    buffer << file.rdbuf();
    file.close();
    return buffer.str();
}

void CreateShaderFromSource(const GPUContext& gpu, const GPUBuffers& buffs, ComputeShader* cs,
                            const char* entryPoint, const std::string& path,
                            const std::string& csLabel,
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
        [&](wgpu::CompilationInfoRequestStatus status, wgpu::CompilationInfo const* info) {
            for (size_t i = 0; i < info->messageCount; ++i) {
                const wgpu::CompilationMessage& message = info->messages[i];
                if (message.type == wgpu::CompilationMessageType::Error) {
                    std::cerr << "Shader compilation error: "
                              << std::string(message.message.data, message.message.length)
                              << std::endl;
                } else if (message.type == wgpu::CompilationMessageType::Warning) {
                    std::cerr << "Shader compilation warning: "
                              << std::string(message.message.data, message.message.length)
                              << std::endl;
                }
            }
            promise.set_value();
        });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
    GetComputeShaderPipeline(gpu.device, buffs, cs, entryPoint, mod, csLabel);
}

void GetAllShaders(const GPUContext& gpu, const GPUBuffers& buffs, Shaders* shaders) {
    std::vector<std::string> empty;
    CreateShaderFromSource(gpu, buffs, &shaders->initPass, "initPass", "Shaders/init.wgsl",
                           "InitPass", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->initBatch, "initBatch", "Shaders/init.wgsl",
                           "InitBatch", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->reduce, "reduce", "Shaders/rts.wgsl", "Reduce",
                           empty);

    CreateShaderFromSource(gpu, buffs, &shaders->spineScan, "spine_scan", "Shaders/rts.wgsl",
                           "Spine Scan", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->downsweep, "downsweep", "Shaders/rts.wgsl",
                           "Downsweep", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdl, "main", "Shaders/csdl.wgsl", "CSDL", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdldf, "main", "Shaders/csdldf.wgsl", "CSDLDF",
                           empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdldfOcc, "main",
                           "Shaders/TestVariants/csdldf_occ.wgsl", "CSDLDF OCC", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->memcpy, "main", "Shaders/memcpy.wgsl", "Memcpy",
                           empty);

    CreateShaderFromSource(gpu, buffs, &shaders->validate, "main", "Shaders/validate.wgsl",
                           "Validate", empty);

    CreateShaderFromSource(gpu, buffs, &shaders->csdldfSimulate, "main",
                           "Shaders/TestVariants/csdldf_simulate.wgsl", "CSDLDF Simulation", empty);
}

void SetComputePass(const ComputeShader& cs, wgpu::CommandEncoder* comEncoder, uint32_t workTiles) {
    wgpu::ComputePassDescriptor comDesc = {};
    comDesc.label = cs.label.c_str();
    wgpu::ComputePassEncoder pass = (*comEncoder).BeginComputePass(&comDesc);
    pass.SetPipeline(cs.computePipeline);
    pass.SetBindGroup(0, cs.bindGroup);
    pass.DispatchWorkgroups(workTiles, 1, 1);
    pass.End();
}

void SetComputePassTimed(const ComputeShader& cs, wgpu::CommandEncoder* comEncoder,
                         const wgpu::QuerySet& querySet, uint32_t workTiles,
                         uint32_t timeStampOffset) {
    wgpu::PassTimestampWrites timeStamp = {};
    timeStamp.beginningOfPassWriteIndex = timeStampOffset * 2;
    timeStamp.endOfPassWriteIndex = timeStampOffset * 2 + 1;
    timeStamp.querySet = querySet;
    wgpu::ComputePassDescriptor comDesc = {};
    comDesc.label = cs.label.c_str();
    comDesc.timestampWrites = &timeStamp;
    wgpu::ComputePassEncoder pass = (*comEncoder).BeginComputePass(&comDesc);
    pass.SetPipeline(cs.computePipeline);
    pass.SetBindGroup(0, cs.bindGroup);
    pass.DispatchWorkgroups(workTiles, 1, 1);
    pass.End();
}

void QueueSync(const GPUContext& gpu) {
    std::promise<void> promise;
    gpu.queue.OnSubmittedWorkDone(wgpu::CallbackMode::AllowProcessEvents,
                                  [&](wgpu::QueueWorkDoneStatus status) {
                                      if (status != wgpu::QueueWorkDoneStatus::Success) {
                                          std::cerr << "uh oh" << std::endl;
                                      }
                                      promise.set_value();
                                  });
    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

void CopyBufferSync(const GPUContext& gpu, wgpu::Buffer* srcReadback, wgpu::Buffer* dstReadback,
                    uint64_t sourceOffsetBytes, uint64_t copySizeBytes) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Copy Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
    comEncoder.CopyBufferToBuffer(*srcReadback, sourceOffsetBytes, *dstReadback, 0ULL,
                                  copySizeBytes);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(1, &comBuffer);
    QueueSync(gpu);
}

template <typename T>
void ReadbackSync(const GPUContext& gpu, wgpu::Buffer* dstReadback, std::vector<T>* readOut,
                  uint64_t readbackSizeBytes) {
    std::promise<void> promise;
    dstReadback->MapAsync(
        wgpu::MapMode::Read, 0, readbackSizeBytes, wgpu::CallbackMode::AllowProcessEvents,
        [&](wgpu::MapAsyncStatus status, wgpu::StringView) {
            if (status == wgpu::MapAsyncStatus::Success) {
                const void* data = dstReadback->GetConstMappedRange(0, readbackSizeBytes);
                std::memcpy(readOut->data(), data, readbackSizeBytes);
                dstReadback->Unmap();
            } else {
                std::cerr << "Bad readback" << std::endl;
            }
            promise.set_value();
        });

    std::future<void> future = promise.get_future();
    while (future.wait_for(std::chrono::nanoseconds(100)) == std::future_status::timeout) {
        gpu.instance.ProcessEvents();
    }
}

template <typename T>
void CopyAndReadbackSync(const GPUContext& gpu, wgpu::Buffer* srcReadback,
                         wgpu::Buffer* dstReadback, std::vector<T>* readOut, uint32_t sourceOffset,
                         uint32_t readbackSize) {
    CopyBufferSync(gpu, srcReadback, dstReadback, sourceOffset * sizeof(T),
                   readbackSize * sizeof(T));
    ReadbackSync(gpu, dstReadback, readOut, readbackSize * sizeof(T));
}

bool Validate(const GPUContext& gpu, GPUBuffers* buffs, const ComputeShader& validate) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Validate Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
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

void ReadbackAndPrintSync(const GPUContext& gpu, GPUBuffers* buffs, uint32_t readbackSize) {
    std::vector<uint32_t> readOut(readbackSize);
    CopyAndReadbackSync(gpu, &buffs->scanOut, &buffs->readback, &readOut, 0, readbackSize);
    for (uint32_t i = 0; i < (readbackSize + 31) / 32; ++i) {
        for (uint32_t k = 0; k < 32; ++k) {
            std::cout << readOut[i * 32 + k] << ", ";
        }
        std::cout << std::endl;
    }
}

void ResolveTimestampQuerys(GPUBuffers* buffs, const wgpu::QuerySet& query,
                            wgpu::CommandEncoder* comEncoder, uint32_t queryEntriesToResolve) {
    (*comEncoder).ResolveQuerySet(query, 0, queryEntriesToResolve, buffs->timestamp, 0ULL);
    (*comEncoder)
        .CopyBufferToBuffer(buffs->timestamp, 0ULL, buffs->readbackTimestamp, 0ULL,
                            queryEntriesToResolve * sizeof(uint64_t));
}

void InitializeUniforms(const GPUContext& gpu, GPUBuffers* buffs, uint32_t size, uint32_t workTiles,
                        uint32_t simulateMask) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Initialize Uniforms Command Encoder";
    wgpu::CommandEncoder comEncoder = gpu.device.CreateCommandEncoder(&comEncDesc);
    std::vector<uint32_t> info{size, (size + 3) / 4, workTiles, simulateMask};
    gpu.queue.WriteBuffer(buffs->info, 0ULL, info.data(), info.size() * sizeof(uint32_t));
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    gpu.queue.Submit(0, &comBuffer);
    QueueSync(gpu);
}

void RTS(const TestArgs& args, wgpu::CommandEncoder* comEncoder, uint32_t baseQueryIndex) {
    if (args.shouldTime) {
        // Each pass gets its own offset from the base for this specific RTS
        // run.
        SetComputePassTimed(args.shaders.reduce, comEncoder, args.gpu.querySet, args.workTiles,
                            baseQueryIndex + 0);
        SetComputePassTimed(args.shaders.spineScan, comEncoder, args.gpu.querySet, 1,
                            baseQueryIndex + 1);
        SetComputePassTimed(args.shaders.downsweep, comEncoder, args.gpu.querySet, args.workTiles,
                            baseQueryIndex + 2);
    } else {
        SetComputePass(args.shaders.reduce, comEncoder, args.workTiles);
        SetComputePass(args.shaders.spineScan, comEncoder, 1);
        SetComputePass(args.shaders.downsweep, comEncoder, args.workTiles);
    }
}

void CSDL(const TestArgs& args, wgpu::CommandEncoder* comEncoder, uint32_t baseQueryIndex) {
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdl, comEncoder, args.gpu.querySet, args.workTiles,
                            baseQueryIndex);
    } else {
        SetComputePass(args.shaders.csdl, comEncoder, args.workTiles);
    }
}

void CSDLDF(const TestArgs& args, wgpu::CommandEncoder* comEncoder, uint32_t baseQueryIndex) {
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldf, comEncoder, args.gpu.querySet, args.workTiles,
                            baseQueryIndex);
    } else {
        SetComputePass(args.shaders.csdldf, comEncoder, args.workTiles);
    }
}

void CSDLDFSimulate(const TestArgs& args, wgpu::CommandEncoder* comEncoder,
                    uint32_t baseQueryIndex) {
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.csdldfSimulate, comEncoder, args.gpu.querySet,
                            args.workTiles, baseQueryIndex);
    } else {
        SetComputePass(args.shaders.csdldfSimulate, comEncoder, args.workTiles);
    }
}

void Memcpy(const TestArgs& args, wgpu::CommandEncoder* comEncoder, uint32_t baseQueryIndex) {
    if (args.shouldTime) {
        SetComputePassTimed(args.shaders.memcpy, comEncoder, args.gpu.querySet, args.workTiles,
                            baseQueryIndex);
    } else {
        SetComputePass(args.shaders.memcpy, comEncoder, args.workTiles);
    }
}

uint32_t GetOccupancySync(const TestArgs& args) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Command Encoder";
    wgpu::CommandEncoder comEncoder = args.gpu.device.CreateCommandEncoder(&comEncDesc);
    SetComputePass(args.shaders.initBatch, &comEncoder, 256);
    SetComputePass(args.shaders.initPass, &comEncoder, 256);
    SetComputePass(args.shaders.csdldfOcc, &comEncoder, args.workTiles);
    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    args.gpu.queue.Submit(1, &comBuffer);
    QueueSync(args.gpu);
    std::vector<uint32_t> readOut(1);
    CopyAndReadbackSync(args.gpu, &args.buffs.misc, &args.buffs.readback, &readOut, 1, 1);
    std::cout << std::endl;
    std::cout << "Estimated CSDLDF Occupancy: " << readOut[0] << std::endl;
    return readOut[0];
}

void RecordToCSV(const TestArgs& args, const DataStruct& data, const std::string& filename) {
    std::ofstream file(filename + ".csv");

    if (args.shouldGetStats) {
        // Write full headers
        file << "time,totalSpins,lookbackLength,fallbacksInitiated,successfulInsertions\n";

        // Write full data
        size_t rows = data.time.size();
        for (size_t i = 0; i < rows; ++i) {
            file << data.time[i] << "," << data.totalSpins[i] << "," << data.lookbackLength[i]
                 << "," << data.fallbacksInitiated[i] << "," << data.successfulInsertions[i]
                 << "\n";
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

void ProcessBatchResults(const TestArgs& args, const std::vector<uint64_t>& batchTimestamps,
                         const std::vector<uint32_t>& batchShaderStats, uint32_t runsThisGPUBatch,
                         uint32_t runsExecutedCount, uint32_t kernelLaunches, TestStatistics& stats,
                         DataStruct& data) {
    for (uint32_t k = 0; k < runsThisGPUBatch; ++k) {
        uint64_t totalRuntimeForThisRun = 0;
        for (uint32_t launch = 0; launch < kernelLaunches; ++launch) {
            uint32_t baseIndex = (k * kernelLaunches + launch) * 2;
            if (baseIndex + 1 < batchTimestamps.size()) {
                totalRuntimeForThisRun +=
                    batchTimestamps[baseIndex + 1] - batchTimestamps[baseIndex];
            }
        }

        stats.totalTime += totalRuntimeForThisRun;
        stats.maxTime = std::max(totalRuntimeForThisRun, stats.maxTime);
        stats.minTime = std::min(totalRuntimeForThisRun, stats.minTime);
        stats.timeMap[totalRuntimeForThisRun]++;

        if (args.shouldGetStats) {
            uint32_t statsBaseIndex = k * args.miscStride;
            double currentSpins =
                static_cast<double>(batchShaderStats[statsBaseIndex]) / args.workTiles;
            uint32_t currentFallbacks = batchShaderStats[statsBaseIndex + 1];
            uint32_t currentInsertions = batchShaderStats[statsBaseIndex + 2];
            double currentLookback =
                static_cast<double>(batchShaderStats[statsBaseIndex + 3]) / args.workTiles;;

            stats.totalSpins += currentSpins;
            stats.totalFallbacksInitiated += currentFallbacks;
            stats.totalSuccessfulInsertions += currentInsertions;
            stats.totalLookbackLength += currentLookback;

            if (args.shouldRecord) {
                uint32_t runId = runsExecutedCount + k;
                data.time[runId] = static_cast<double>(totalRuntimeForThisRun);
                data.totalSpins[runId] = currentSpins;
                data.fallbacksInitiated[runId] = currentFallbacks;
                data.successfulInsertions[runId] = currentInsertions;
                data.lookbackLength[runId] = currentLookback;
            }
        } else if (args.shouldRecord) {
            uint32_t runId = runsExecutedCount + k;
            data.time[runId] = static_cast<double>(totalRuntimeForThisRun);
        }
    }
}

void PrintSummary(const std::string& testLabel, const TestStatistics& stats, const TestArgs& args,
                  uint32_t maxRunsInGPUBatch) {
    printf("\n--- Timing Summary for: %s (Warmups: %u, Batch: %u/%u) ---\n", testLabel.c_str(),
           args.warmupRuns, args.batchSize, maxRunsInGPUBatch);
    if (args.totalRuns == 0) {
        printf("No timed tests were run to analyze.\n");
        printf("---------------------------------------------------\n");
        return;
    }

    printf("Statistics for %u timed test run(s) in batches of %u:\n", args.totalRuns,
           maxRunsInGPUBatch);

    double totalTimeAllSec = static_cast<double>(stats.totalTime) / 1e9;
    printf("  Total time: %.4f s\n", totalTimeAllSec);

    double avgTimeRunNS = (stats.totalTime > 0 && args.totalRuns > 0)
                              ? (static_cast<double>(stats.totalTime) / args.totalRuns)
                              : 0.0;
    printf("  Average time per run: %.0f ns\n", avgTimeRunNS);

    if (stats.minTime != ~0ULL) {
        printf("  Min time per run: %llu ns\n", stats.minTime);
    } else {
        printf("  Min time per run: N/A (no tests run or all had issues)\n");
    }
    printf("  Max time per run: %llu ns\n", stats.maxTime);

    std::multimap<unsigned int, uint64_t> reverseTimeMap;
    for (const auto& pair : stats.timeMap) {
        reverseTimeMap.insert({pair.second, pair.first});
    }

    int topNToPrint = std::min(5, (int)reverseTimeMap.size());
    printf("  Top %d runtimes (runtime ns => number of runs):\n", topNToPrint);
    if (reverseTimeMap.empty()) {
        printf("    { No distinct runtimes recorded }\n");
    } else {
        printf("    { ");
        int numPrinted = 0;
        for (auto it = reverseTimeMap.rbegin();
             it != reverseTimeMap.rend() && numPrinted < topNToPrint; ++it) {
            if (numPrinted > 0) {
                printf(", ");
            }
            printf("%llu => %u", it->second, it->first);
            numPrinted++;
        }
        printf(" }\n");
    }

    if (totalTimeAllSec > 0) {
        double throughputElePerSec =
            (static_cast<double>(args.size) * args.totalRuns) / totalTimeAllSec;
        printf("  Estimated speed (on %u elements/run): %.2e ele/s\n", args.size,
               throughputElePerSec);
    } else {
        printf("  Estimated speed: N/A\n");
    }

    if (args.shouldGetStats) {
        printf("\n--- Shader Statistics Summary ---\n");
        printf("  Avg spins per workgroup per run: %.2f\n",
               static_cast<double>(stats.totalSpins) / args.totalRuns);
        printf("  Avg lookback length per workgroup per run: %.2f\n",
               static_cast<double>(stats.totalLookbackLength) / args.totalRuns);
        printf("  Avg fallbacks initiated per run: %.2f\n",
               static_cast<double>(stats.totalFallbacksInitiated) /
                   static_cast<double>(args.totalRuns));
        printf("  Avg successful fallback insertions per run: %.2f\n",
               static_cast<double>(stats.totalSuccessfulInsertions) /
                   static_cast<double>(args.totalRuns));
    }
    printf("---------------------------------------------------\n");
}

void RunWarmup(const TestArgs& args,
               void (*Pass)(const TestArgs&, wgpu::CommandEncoder*, uint32_t)) {
    wgpu::CommandEncoderDescriptor comEncDesc = {};
    comEncDesc.label = "Warmup Command Encoder";
    wgpu::CommandEncoder comEncoder = args.gpu.device.CreateCommandEncoder(&comEncDesc);

    for (uint32_t i = 0; i < args.warmupRuns; ++i) {
        SetComputePass(args.shaders.initPass, &comEncoder, 256);
        (*Pass)(args, &comEncoder, /*baseQueryOffset=*/0);  // not actually timing, so dont care.
    }

    wgpu::CommandBuffer comBuffer = comEncoder.Finish();
    args.gpu.queue.Submit(1, &comBuffer);
    QueueSync(args.gpu);
}

void Run(std::string testLabel, const TestArgs& args,
         void (*Pass)(const TestArgs&, wgpu::CommandEncoder*, uint32_t), uint32_t kernelLaunches) {
    if (args.totalRuns == 0) {
        printf("%s: No actual tests to run (totalRuns is 0).\n", testLabel.c_str());
        return;
    }

    if (args.warmupRuns) {
        RunWarmup(args, (*Pass));
    }

    TestStatistics stats;
    DataStruct data(args);
    std::vector<uint32_t> batchShaderStats(args.maxQueryEntries / 2 * args.miscStride);
    std::vector<uint64_t> batchTimestamps(args.maxQueryEntries);

    const uint32_t maxRunsInGPUBatch =
        std::min(args.maxQueryEntries / (kernelLaunches * 2), args.batchSize);
    uint32_t runsExecutedCount = 0;

    while (runsExecutedCount < args.totalRuns) {
        uint32_t runsThisGPUBatch = std::min(args.totalRuns - runsExecutedCount, maxRunsInGPUBatch);
        uint32_t queriesThisGPUBatch = runsThisGPUBatch * kernelLaunches * 2;

        wgpu::CommandEncoderDescriptor comEncDesc = {};
        wgpu::CommandEncoder comEncoder = args.gpu.device.CreateCommandEncoder(&comEncDesc);

        SetComputePass(args.shaders.initBatch, &comEncoder, 256);
        for (uint32_t j = 0; j < runsThisGPUBatch; ++j) {
            SetComputePass(args.shaders.initPass, &comEncoder, 256);
            (*Pass)(args, &comEncoder, j* kernelLaunches);
        }
        ResolveTimestampQuerys(&args.buffs, args.gpu.querySet, &comEncoder, queriesThisGPUBatch);

        wgpu::CommandBuffer comBuffer = comEncoder.Finish();
        args.gpu.queue.Submit(1, &comBuffer);
        QueueSync(args.gpu);

        ReadbackSync(args.gpu, &args.buffs.readbackTimestamp, &batchTimestamps,
                     queriesThisGPUBatch * sizeof(uint64_t));
        if (args.shouldGetStats) {
            CopyAndReadbackSync(args.gpu, &args.buffs.misc, &args.buffs.readback, &batchShaderStats,
                                args.miscStride, runsThisGPUBatch * args.miscStride);
        }

        ProcessBatchResults(args, batchTimestamps, batchShaderStats, runsThisGPUBatch,
                            runsExecutedCount, kernelLaunches, stats, data);

        runsExecutedCount += runsThisGPUBatch;
    }

    if (args.shouldValidate && !Validate(args.gpu, &args.buffs, args.shaders.validate)) {
        printf("Validation Failed, timing data is suspect, ending early.\n");
        return;
    }

    PrintSummary(testLabel, stats, args, maxRunsInGPUBatch);

    if (args.shouldRecord) {
        RecordToCSV(args, data, testLabel);
    }
}

void TestMemcpy(std::string deviceName, const TestArgs& args) {
    TestArgs memcpyArgs = args;
    memcpyArgs.shouldValidate = false;
    Run(deviceName + "Memcpy", memcpyArgs, Memcpy, 1);
}

void TestRTS(std::string deviceName, const TestArgs& args) {
    Run(deviceName + "RTS", args, RTS, 3);
}

void TestCSDL(std::string deviceName, const TestArgs& args) {
    Run(deviceName + "CSDL", args, CSDL, 1);
}

void TestCSDLDF(std::string deviceName, const TestArgs& args) {
    DataStruct data(args);
    GetOccupancySync(args);
    Run(deviceName + "CSDLDF", args, CSDLDF, 1);
}

void TestFull(std::string deviceName, uint32_t MAX_SIMULATE, const TestArgs& args) {
    Run(deviceName + "RTS", args, RTS, 3);
    GetOccupancySync(args);
    Run(deviceName + "CSDLDF", args, CSDLDF, 1);
    TestArgs simArgs = args;
    simArgs.shouldGetStats = true;
    InitializeUniforms(simArgs.gpu, &simArgs.buffs, simArgs.size, simArgs.workTiles, 0xffffffff);
    Run(deviceName + "CSDLDF_Stats", simArgs, CSDLDFSimulate, 1);
    for (uint32_t i = 0; i <= MAX_SIMULATE; ++i) {
        uint32_t mask = (1 << i) - 1;
        InitializeUniforms(simArgs.gpu, &simArgs.buffs, simArgs.size, simArgs.workTiles, mask);
        Run(deviceName + "CSDLDF_" + std::to_string(1 << i), simArgs, CSDLDFSimulate, 1);
    }
}

void TestSize(std::string deviceName, uint32_t PART_SIZE, const TestArgs& args) {
    const uint32_t minPow = 10;
    const uint32_t maxPow = 25;

    for (uint32_t i = minPow; i <= maxPow; ++i) {
        uint32_t currentSize = 1u << i;
        uint32_t currentWorkTiles = (currentSize + PART_SIZE - 1) / PART_SIZE;
        TestArgs localArgs = args;
        localArgs.size = currentSize;
        localArgs.workTiles = currentWorkTiles;
        InitializeUniforms(localArgs.gpu, &localArgs.buffs, currentSize, currentWorkTiles, 0);
        std::string testLabel = deviceName + "CSDLDF_Size_" + std::to_string(currentSize);
        Run(testLabel, localArgs, CSDLDF, 1);
    }
}

void TestMemcpySize(std::string deviceName, uint32_t PART_SIZE, const TestArgs& args) {
    const uint32_t minPow = 10;
    const uint32_t maxPow = 25;

    for (uint32_t i = minPow; i <= maxPow; ++i) {
        uint32_t currentSize = 1u << i;
        uint32_t currentWorkTiles = (currentSize + PART_SIZE - 1) / PART_SIZE;
        TestArgs memcpyArgs = args;
        memcpyArgs.size = currentSize;
        memcpyArgs.workTiles = currentWorkTiles;
        memcpyArgs.shouldValidate = false;  // Memcpy test doesn't need validation
        InitializeUniforms(memcpyArgs.gpu, &memcpyArgs.buffs, currentSize, currentWorkTiles, 0);
        std::string testLabel = deviceName + "Memcpy_Size_" + std::to_string(currentSize);
        Run(testLabel, memcpyArgs, Memcpy, 1);
    }
}

struct CommandLineArgs {
    TestType testType;
    uint32_t warmupRuns;
    uint32_t totalRuns;
    uint32_t batchSize;
    uint32_t sizeExponent;
    bool shouldRecord;
    std::string deviceName;
};
void ParseArguments(int argc, char* argv[], CommandLineArgs& args) {
    auto printUsage = []() {
        std::cerr
            << "Usage: <TestType> [options]\n\n" // Removed [deviceName] from here
            << "  <TestType>: rts | csdl | csdldf | full | sizecsdldf | sizememcpy\n\n"
            << "Options:\n"
            << "  --label <name>       Optional name to prefix output files (e.g., 'MyCoolGPU').\n"
            << "  --record             Enable recording to CSV.\n"
            << "  --runs <N>           Set the number of timed runs (default: 4192).\n"
            << "  --warmup <N>         Set the number of warmup runs (default: 1000).\n"
            << "  --batch <N>          Set the max batch size per submission (default: 2048).\n"
            << "  --size <N>           Set the problem size as a power of 2 (i.e., 1 << N) "
               "(default: 25).\n"
            << "  -h, --help           Show this help message.\n";
            // Removed the separate [deviceName] description
    };

    if (argc < 2) {
        printUsage();
        throw std::invalid_argument("A test type is required.");
    }

    std::string testTypeStr = argv[1];
    if (testTypeStr == "rts")
        args.testType = Rts;
    else if (testTypeStr == "csdl")
        args.testType = Csdl;
    else if (testTypeStr == "csdldf")
        args.testType = Csdldf;
    else if (testTypeStr == "full")
        args.testType = Full;
    else if (testTypeStr == "sizecsdldf")
        args.testType = SizeCsdldf;
    else if (testTypeStr == "sizememcpy")
        args.testType = SizeMemcpy;
    else if (testTypeStr == "--help" || testTypeStr == "-h") {
        printUsage();
        exit(EXIT_SUCCESS);
    } else {
        throw std::invalid_argument("Invalid TestType: " + testTypeStr);
    }

    std::vector<std::string> arguments;
    for (int i = 2; i < argc; ++i) {
        arguments.push_back(argv[i]);
    }

    for (size_t i = 0; i < arguments.size(); ++i) {
        std::string arg = arguments[i];
        if (arg == "--record") {
            args.shouldRecord = true;
        } else if (arg == "--runs") {
            if (i + 1 < arguments.size()) {
                args.totalRuns = std::stoul(arguments[++i]);
            } else {
                throw std::invalid_argument("--runs requires a value.");
            }
        } else if (arg == "--warmup") {
            if (i + 1 < arguments.size()) {
                args.warmupRuns = std::stoul(arguments[++i]);
            } else {
                throw std::invalid_argument("--warmup requires a value.");
            }
        } else if (arg == "--batch") {
            if (i + 1 < arguments.size()) {
                args.batchSize = std::stoul(arguments[++i]);
            } else {
                throw std::invalid_argument("--batch requires a value.");
            }
        } else if (arg == "--size") {
            if (i + 1 < arguments.size()) {
                args.sizeExponent = std::stoul(arguments[++i]);
            } else {
                throw std::invalid_argument("--size requires a value.");
            }
        } else if (arg == "--label") {
            if (i + 1 < arguments.size()) {
                args.deviceName = arguments[++i] + "_";
            } else {
                throw std::invalid_argument("--label requires a value.");
            }
        }
        else {
            // Any argument that is not a recognized flag is now an error.
            throw std::invalid_argument("Unknown option: " + arg);
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        CommandLineArgs cliArgs;
        cliArgs.warmupRuns = 1000;
        cliArgs.totalRuns = 4192;
        cliArgs.batchSize = 2048;
        cliArgs.sizeExponent = 25;
        cliArgs.shouldRecord = false;
        cliArgs.deviceName = "";
        ParseArguments(argc, argv, cliArgs);

        constexpr uint32_t MISC_STRIDE = 4;
        constexpr uint32_t PART_SIZE = 4096;
        constexpr uint32_t MAX_QUERY_ENTRIES = 4096;
        constexpr uint32_t MAX_READBACK_SIZE = 16384;
        constexpr uint32_t MAX_SIMULATE = 9;

        const uint32_t size = 1u << cliArgs.sizeExponent;
        const uint32_t workTiles = (size + PART_SIZE - 1) / PART_SIZE;
        const uint32_t readbackSize = 256;

        GPUContext gpu;
        GPUBuffers buffs;
        Shaders shaders;
        GetGPUContext(&gpu, MAX_QUERY_ENTRIES);
        GetGPUBuffers(gpu.device, &buffs, ((1 << 25) + PART_SIZE - 1) / PART_SIZE,
                      MAX_QUERY_ENTRIES, (1 << 25), MISC_STRIDE, MAX_READBACK_SIZE);
        GetAllShaders(gpu, buffs, &shaders);

        TestArgs args = {gpu,
                         buffs,
                         shaders,
                         size,
                         workTiles,
                         readbackSize,
                         cliArgs.warmupRuns,
                         cliArgs.totalRuns,
                         cliArgs.batchSize,
                         MAX_QUERY_ENTRIES,
                         MISC_STRIDE,
                         /*shouldValidate=*/true,
                         /*shouldReadback=*/false,
                         /*shouldTime=*/true,
                         /*shouldGetStats=*/false,
                         cliArgs.shouldRecord};

        InitializeUniforms(gpu, &buffs, size, workTiles, 0);

        switch (cliArgs.testType) {
            case Rts:
                TestRTS(cliArgs.deviceName, args);
                break;
            case Csdl:
                TestCSDL(cliArgs.deviceName, args);
                break;
            case Csdldf:
                TestCSDLDF(cliArgs.deviceName, args);
                break;
            case Full:
                TestMemcpy(cliArgs.deviceName, args);
                TestFull(cliArgs.deviceName, MAX_SIMULATE, args);
                break;
            case SizeCsdldf:
                TestSize(cliArgs.deviceName, PART_SIZE, args);
                break;
            case SizeMemcpy:
                TestMemcpySize(cliArgs.deviceName, PART_SIZE, args);
                break;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}