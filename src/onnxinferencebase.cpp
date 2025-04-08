#include "../include/onnxinferencebase.hpp"
#include <iostream>

OnnxInferenceBase::OnnxInferenceBase() : m_env(ORT_LOGGING_LEVEL_WARNING, "Default") {
    std::cout << "Initializing ONNX Runtime environment" << std::endl;
}

OnnxInferenceBase::~OnnxInferenceBase() {
    std::cout << "Cleaning up ONNX Runtime resources" << std::endl;
}

void OnnxInferenceBase::ConfigureSession(bool use_cuda)
{
    m_session_options.SetIntraOpNumThreads(1);
    m_session_options.SetInterOpNumThreads(1);

    // Optimization will take time and memory during startup
    m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

    // Configure CUDA if requested
    if (use_cuda)
    {
        ConfigureCuda();
    }
} 

void OnnxInferenceBase::ConfigureCuda()
{
    std::cout << "Configuring CUDA" << std::endl;
    m_cuda_options.device_id = 0;
    m_cuda_options.arena_extend_strategy = 0;
    m_cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    m_cuda_options.do_copy_in_default_stream = 0;
    m_session_options.AppendExecutionProvider_CUDA(m_cuda_options);
}

bool OnnxInferenceBase::LoadModel(const std::string& model_path)
{

    // Loading the model
    try {
        std::cout << "Loading model from " << model_path << std::endl;
        m_session = Ort::Session(m_env, model_path.c_str(), m_session_options);
        std::cout << "Model loaded successfully" << std::endl;
    } 
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl << ", Code: " << e.GetOrtErrorCode() << std::endl;
        return false;
    }

    // Allocate memory for input tensor
    try {
        m_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl << ", Code: " << e.GetOrtErrorCode() << std::endl;
        return false;
    }

    return true;
}

void OnnxInferenceBase::SetInputNodeNames(std::vector<const char*>* names) {
    if (names) {
        m_inputNodeNames = *names;
    }
}

void OnnxInferenceBase::SetOutputNodeNames(std::vector<const char*>* names) {
    if (names) {
        m_outputNodeNames = *names;
    }
}

void OnnxInferenceBase::SetInputDemensions(std::vector<int64_t> dims) {
    m_inputNodeDims = dims;
}


