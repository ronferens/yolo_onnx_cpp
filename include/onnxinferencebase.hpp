#ifndef ONNXINFERENCEBASE_HPP
#define ONNXINFERENCEBASE_HPP

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>


class OnnxInferenceBase {
    public:
        OnnxInferenceBase();
        ~OnnxInferenceBase();
        void ConfigureSession(bool use_cuda);
        void ConfigureCuda();
        bool LoadModel(const std::string& model_path);
        void SetInputNodeNames(std::vector<const char*>* input_node_names);
        void SetInputDemensions(std::vector<int64_t> input_node_dims);
        void SetOutputNodeNames(std::vector<const char*>* output_node_names);

    protected:
        Ort::Session m_session{nullptr};
        Ort::Env m_env;
        Ort::SessionOptions m_session_options;
        OrtCUDAProviderOptions m_cuda_options;
        Ort::MemoryInfo m_memory_info{nullptr};
        std::vector<const char*> m_outputNodeNames;    // output node names
        std::vector<const char*> m_inputNodeNames;     // Input node names
        std::vector<int64_t> m_inputNodeDims;          // Input node dimension
};

#endif
