cmake_minimum_required(VERSION 3.10)

project(model-explorer)

include_directories(/home/shlok/osprey/onnx/onnxruntime-linux-x64-1.21.0/include include)
link_directories(/home/shlok/osprey/onnx/onnxruntime-linux-x64-1.21.0/lib)

add_executable(model-explorer main.cpp src/inference.cpp)
target_link_libraries(model-explorer PRIVATE onnxruntime)
# target_link_libraries(model-explorer PRIVATE onnxruntime)
# target_include_directories(model-explorer PUBLIC /home/shlok/osprey/onnx/onnxruntime-linux-x64-1.21.0)