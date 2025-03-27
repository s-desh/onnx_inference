// #include "/home/shlok/osprey/onnx/onnxruntime-linux-x64-1.21.0/include/onnxruntime_cxx_api.h"
// #include <algorithm>  // std::generate
// #include <cassert>
// #include <cstddef>
// #include <cstdint>
// #include <iostream>
// #include <sstream>
// #include <string>
// #include <vector>
#include "inference.h"


// int main()
// {
//     // Load the model and create InferenceSession
//     Ort::Env env;
//     std::string model_path = "/home/shlok/osprey/onnx/policy.onnx";
//     Ort::Session session(env, model_path, Ort::SessionOptions{ nullptr });
//     // Load and preprocess the input image to inputTensor
//     ...
//     // Run inference
//     std::vector outputTensors =
//     session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, 
//     inputNames.size(), outputNames.data(), outputNames.size());
//     const float* outputDataPtr = outputTensors[0].GetTensorMutableData();
//     std::cout << outputDataPtr[0] << std::endl;
// }
// std::string print_shape(const std::vector<std::int64_t>& v) {
//     std::stringstream ss("");
//     for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
//     ss << v[v.size() - 1];
//     return ss.str();
//   }
  
//   int calculate_product(const std::vector<std::int64_t>& v) {
//     int total = 1;
//     for (auto& i : v) total *= i;
//     return total;
//   }
  
//   template <typename T>
//   Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
//     Ort::MemoryInfo mem_info =
//         Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//     auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
//     return tensor;
//   }
  

// int main(int argc, ORTCHAR_T* argv[]) {
//     if (argc != 2) {
//         std::cout << "Usage: ./onnx-api-example <onnx_model.onnx>" << std::endl;
//         return -1;
//     }

//     std::basic_string<ORTCHAR_T> model_file = argv[1];

//     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
//     Ort::SessionOptions session_options;
//     Ort::Session session = Ort::Session(env, model_file.c_str(), session_options);

//     // print name/shape of inputs
//     Ort::AllocatorWithDefaultOptions allocator;
//     std::vector<std::string> input_names;
//     std::vector<std::int64_t> input_shapes;
//     std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
//     for (std::size_t i = 0; i < session.GetInputCount(); i++) {
//         input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
//         input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
//         std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
//     }
//     // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
//     for (auto& s : input_shapes) {
//         if (s < 0) {
//         s = 1;
//         }
//     }

//     // print name/shape of outputs
//     std::vector<std::string> output_names;
//     std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
//     for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
//         output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
//         auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
//         std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
//     }

//     // assert(input_names.size() == 1 && output_names.size() == 1);

//         // Create a single Ort tensor of random numbers
//     auto input_shape = input_shapes;
//     auto total_number_elements = calculate_product(input_shape);

//     // generate random numbers in the range [0, 255]
//     std::vector<float> input_tensor_values = {-0.27438995, -0.12135763, -0.21477927, -0.08903223,  0.2935632 ,
//         -0.4193306 , -0.07753016, -0.17574848, -0.29855108, -0.32094336,
//         -0.1390053 , -0.2912455 ,  0.31930816, -0.14087664,  0.2285923 ,
//         -0.01238768};
//     // std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });
//     std::vector<Ort::Value> input_tensors;
//     input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_shape));

//     // double-check the dimensions of the input tensor
//     assert(input_tensors[0].IsTensor() && input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
//     std::cout << "\ninput_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

//     // pass data through model
//     std::vector<const char*> input_names_char(input_names.size(), nullptr);
//     std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
//                     [&](const std::string& str) { return str.c_str(); });

//     std::vector<const char*> output_names_char(output_names.size(), nullptr);
//     std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
//                     [&](const std::string& str) { return str.c_str(); });

//     std::cout << "Input count: " << input_names_char.size() << ", Output count: " << output_names_char.size() << std::endl;


//     std::cout << "Running model..." << std::endl;
//     try {
//         auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
//                                         input_names_char.size(), output_names_char.data(), output_names_char.size());
//         std::cout << "Done!" << std::endl;
//         const float* outputDataPtr = output_tensors[0].GetTensorMutableData<float>();
//         // std::cout << outputDataPtr << std::endl;
//         for (size_t i = 0; i < 7; ++i) {
//             std::cout << outputDataPtr[i] << " " << std::endl;
//         }

//         // double-check the dimensions of the output tensors
//         // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
//         assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());
//     } catch (const Ort::Exception& exception) {
//         std::cout << "ERROR running model inference: " << exception.what() << std::endl;
//         exit(-1);
//     }
// }

int main() {

    std::vector<float> running_variance = {0.8103, 0.2566, 0.5439, 3.1260, 1.2836, 0.4474, 0.0355, 0.0173, 0.0116,
        0.1919, 0.1203, 0.6398, 0.0067, 0.0073, 0.0129, 0.0237};
    
    std::vector<float> running_mean = {0.2523,  0.0792,  0.0730, -0.0574, -0.0294,  0.0224, -0.0064, -0.0065,
        -0.9672,  0.1143,  0.0486,  0.2306,  0.9733, -0.0027, -0.0386, -0.0218}; 

    InferenceEngine engine("/home/shlok/osprey/onnx/policy.onnx", running_variance, running_mean);
    std::vector<float> input = {5.2545e-03,  1.7770e-02, -8.5364e-02, -2.1483e-01,  3.0324e-01,
        -2.5810e-01, -2.0970e-02, -2.9584e-02, -9.9934e-01, -2.6246e-02,
         3.7281e-04, -2.3498e-03,  9.9953e-01, -1.4671e-02, -1.2626e-02,
        -2.3741e-02};

    std::vector<float> output = engine.run(input);
    for (size_t i = 0; i < std::min(output.size(), size_t(7)); ++i) {
        std::cout << output[i] << " " << std::endl;
    }
    return 0;

}