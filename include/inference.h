#ifndef INFERENCE_H
#define INFERENCE_H


#include "/home/shlok/osprey/onnx/onnxruntime-linux-x64-1.21.0/include/onnxruntime_cxx_api.h"
#include <algorithm>  // std::generate
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


class InferenceEngine
{
    
    public:
    
        InferenceEngine(std::string model_file, std::vector<float> running_variance, std::vector<float> running_mean);

        std::vector<float> run(const std::vector<float>& input);
        std::vector<float> scale_input(const std::vector<float>& input);
        
        template <typename T>
        Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape);

    
    private:
        std::string model_file_;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        
        std::vector<float> running_variance_ ;
        
        std::vector<float> running_mean_; 
        
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::vector<Ort::Value> input_tensors;
        std::vector<std::int64_t> input_shapes;
        std::vector<const char*> input_names_char;
        std::vector<const char*> output_names_char;
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::Session session;
};

#endif // INFERENCE_H