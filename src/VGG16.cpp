#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneDNN/examples/example_utils.hpp"

#include "VGG16.hpp"

using namespace dnnl;

// CPU engine implementation

/*
During training, the input to our ConvNets is a fixed-size 224 × 224 RGB image. The only pre-
processing we do is subtracting the mean RGB value, computed on the training set, from each pixel.
The image is passed through a stack of convolutional (conv.) layers, where we use filters with a very
small receptive field: 3 × 3 (which is the smallest size to capture the notion of left/right, up/down,
center). In one of the configurations we also utilise 1 × 1 convolution filters, which can be seen as
a linear transformation of the input channels (followed by non-linearity). The convolution stride is
fixed to 1 pixel; the spatial padding of conv. layer input is such that the spatial resolution is preserved
after convolution, i.e. the padding is 1 pixel for 3 × 3 conv. layers. Spatial pooling is carried out by
five max-pooling layers, which follow some of the conv. layers (not all the conv. layers are followed
by max-pooling). Max-pooling is performed over a 2 × 2 pixel window, with stride 2.
*/
// VGG16 D configuration
void VGG16(engine::kind engine_kind){
    using tag = memory::format_tag;
    using dt = memory::data_type;   
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
    const memory::dim padding = 1;
    const memory::dim conv_stride = 1;
    const memory::dim pool_stride = 2;

    // convolutional layer 1: 224x224x64
    memory::dims conv1_src_tz = {batch, input_channels, input_H, input_W};
    memory::dims conv1_weights_tz = {64, input_channels, 3, 3};
    memory::dims conv1_bias_tz = {64};
    memory::dims conv1_dst_tz = {batch, 64, input_H, input_W};
    memory::dims conv1_strides = {conv_stride, conv_stride};
    memory::dims conv1_padding = {padding, padding};

    // Allocate buffers for input data and weights, and create memory descriptors
    std::vector<float> user_src(batch * input_channels * input_H * input_W);
    std::vector<float> user_dst(batch * 64 * input_H * input_W);
    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));

    // Create user memory
    auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    auto user_weights_memory
            = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);
    auto conv1_user_bias_memory
            = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);

    // Create convolution memory descriptors with format_tag::any
    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);

    // Create convolution descriptor
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
        conv1_dst_md, conv1_strides, conv1_padding, conv1_padding);

    // Create convolution primitive descriptor 
    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);

    // Check if data and weights format required by convolution is different 
    // from the user format, if so reorder the memory layout
    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                {DNNL_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
                .execute(s, user_weights_memory, conv1_weights_memory);
    }

    // Create memory for output
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(),eng);

    // Create the convolution primitive
    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
            {DNNL_ARG_BIAS, conv1_user_bias_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});


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