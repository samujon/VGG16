#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "VGG16.hpp"

using namespace dnnl;


// VGG16 D configuration
void VGG16(engine::kind engine_kind){
    // Create an engine
    engine eng(engine_kind, 0);
    stream s(eng);

    //Create network
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Handle one input image at a time
    const memory::dim batch = 1;

    // VGG16: 224x224x3 input image
    const memory::dim input_channels = 3;
    const memory::dim input_H = 224;
    const memory::dim input_W = 224;

    // convolutional layer 1: 224x224x64
    memory::dims conv1_src_tz = {batch, input_channels, input_H, input_W};
    memory::dims conv1_weights_tz = {64, input_channels, 3, 3};

    
    // convolutional layer 2: 224x224x64
    // max pooling layer 1: 112x112x64
    // convolutional layer 3: 112x112x128
    // convolutional layer 4: 112x112x128
    // max pooling layer 2: 56x56x128
    // convolutional layer 5: 56x56x256
    // convolutional layer 6: 56x56x256
    // convolutional layer 7: 56x56x256
    // max pooling layer 3: 28x28x256
    // convolutional layer 8: 28x28x512
    // convolutional layer 9: 28x28x512
    // convolutional layer 10: 28x28x512
    // max pooling layer 4: 14x14x512
    // convolutional layer 11: 14x14x512
    // convolutional layer 12: 14x14x512
    // convolutional layer 13: 14x14x512
    // max pooling layer 5: 7x7x512
    // fully connected layer 1: 4096
    // fully connected layer 2: 4096
    // softmax layer: 1000




}