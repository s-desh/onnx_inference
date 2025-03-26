#include "inference.h"


InferenceEngine::InferenceEngine(std::string model_file, std::vector<float> running_variance, std::vector<float> running_mean) : model_file_(model_file), 
    running_variance_(running_variance), running_mean_(running_mean), session(env, model_file_.c_str(), session_options)
{
    
    
    // Ort::SessionOptions session_options;
    // session = Ort::Session(env, model_file_, Ort::SessionOptions{ nullptr });

    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        // std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }

    for (auto& s : input_shapes) {
        if (s < 0) {
            s = 1;
        }
    }

    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        // std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
    }

    input_names_char = std::vector<const char*>(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
                    [&](const std::string& str) { return str.c_str(); });

    output_names_char = std::vector<const char*>(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
                    [&](const std::string& str) { return str.c_str(); });


}


template <typename T>
Ort::Value InferenceEngine::vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
  return tensor;
}


std::vector<float> InferenceEngine::scale_input(const std::vector<float>& input)
{
    std::vector<float> scaled_input;
    for (size_t i = 0; i < input.size(); ++i) {
        scaled_input.push_back((input[i] - running_mean_[i]) / sqrt(running_variance_[i]) + 1e-8);
        std::cout << scaled_input[i] << " ";
    }
    return scaled_input;
}

std::vector<float> InferenceEngine::run(const std::vector<float>& input)
{
    std::vector<Ort::Value> input_tensors;
    auto scaled_input = scale_input(input);
    input_tensors.emplace_back(vec_to_tensor<float>(scaled_input, input_shapes));




    std::cout << "Input count: " << input_names_char.size() << ", Output count: " << output_names_char.size() << std::endl;

    std::cout << "Running model..." << std::endl;
    try {
        // Move this after the for loop
        // std::cout << "Input count: " << input_names.size() << ", Output count: " << output_names.size() << std::endl;

        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
                                        input_names_char.size(), output_names_char.data(), output_names_char.size());
        std::cout << "Done!" << std::endl;
        const float* outputDataPtr = output_tensors[0].GetTensorMutableData<float>();
        
        
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        size_t output_size = 1;
        for (auto dim : output_shape) output_size *= dim;

        std::vector<float> output(outputDataPtr, outputDataPtr + output_size);

        // for (size_t i = 0; i < size_t(7); ++i) {
        //     std::cout << output[i] << " " << std::endl;
        // }
        
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

        return output;
    } catch (const Ort::Exception& exception) {
        throw std::runtime_error("ERROR running model inference: " + std::string(exception.what()));
    }

}