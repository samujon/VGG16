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
// In comparison with AlexNet, VGG16 does not LRN, local response normalization
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

    // -----------------------------------------------------------
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
        conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding, conv1_padding);

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

    // -----------------------------------------------------------
    // ReLu1
    const float negative1_slope = 0.0f;

    // Create ReLu primitive
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv1_dst_memory.get_desc(),
            negative1_slope);
    auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);

    net.push_back(eltwise_forward(relu1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});

    // -----------------------------------------------------------
    // convolutional layer 2: 224x224x64
    memory::dims conv2_src_tz = {batch, input_channels, input_H, input_W};
    memory::dims conv2_weights_tz = {64, input_channels, 3, 3};
    memory::dims conv2_bias_tz = {64};
    memory::dims conv2_dst_tz = {batch, 64, input_H, input_W};
    memory::dims conv2_strides = {conv_stride, conv_stride};
    memory::dims conv2_padding = {padding, padding};

    // Allocate buffers for input data and weights, and create memory descriptors
    std::vector<float> conv2_weights(product(conv2_weights_tz));
    std::vector<float> conv2_bias(product(conv2_bias_tz));

    // Create user memory
    auto conv2_user_weights_memory
            = memory({{conv2_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv2_weights.data(), conv2_user_weights_memory);
    auto conv2_user_bias_memory
            = memory({{conv2_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv2_bias.data(), conv2_user_bias_memory);

    // Create convolution memory descriptors with format_tag::any
    auto conv2_src_md = memory::desc({conv2_src_tz}, dt::f32, tag::any);
    auto conv2_weights_md = memory::desc({conv2_weights_tz}, dt::f32, tag::any);
    auto conv2_bias_md = memory::desc({conv2_bias_tz}, dt::f32, tag::any);
    auto conv2_dst_md = memory::desc({conv2_dst_tz}, dt::f32, tag::any);

    // Create convolution descriptor
    auto conv2_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv2_src_md, conv2_weights_md,
        conv2_bias_md,conv2_dst_md, conv2_strides, conv2_padding, conv2_padding);

    // Create convolution primitive descriptor 
    auto conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, eng);

    // Check if data and weights format required by convolution is different 
    // from the user format, if so reorder the memory layout
    auto conv2_src_memory = conv1_dst_memory;
    if (conv2_prim_desc.src_desc() != conv2_src_memory.get_desc()) {
        conv2_src_memory = memory(conv2_prim_desc.src_desc(), eng);
        net.push_back(reorder(conv1_dst_memory, conv2_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv1_dst_memory},
                {DNNL_ARG_TO, conv2_src_memory}});
    }

    auto conv2_weights_memory = conv2_user_weights_memory;
    if (conv2_prim_desc.weights_desc() != conv2_user_weights_memory.get_desc()) {
        conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), eng);
        reorder(conv2_user_weights_memory, conv2_weights_memory)
                .execute(s, conv2_user_weights_memory, conv2_weights_memory);
    }

    // Create memory for output
    auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(),eng);

    // Create the convolution primitive
    net.push_back(convolution_forward(conv2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_src_memory},
            {DNNL_ARG_WEIGHTS, conv2_weights_memory},
            {DNNL_ARG_BIAS, conv2_user_bias_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});


    // -----------------------------------------------------------
    // ReLu2
    const float negative2_slope = 0.0f;

    // Create ReLu primitive
    auto relu2_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv2_dst_memory.get_desc(),
            negative2_slope);
    auto relu2_prim_desc = eltwise_forward::primitive_desc(relu2_desc, eng);

    net.push_back(eltwise_forward(relu2_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv2_dst_memory},
            {DNNL_ARG_DST, conv2_dst_memory}});

    // -----------------------------------------------------------
    // max pooling layer 1: 112x112
    // 224x224 -> 112x112
    memory::dims pool1_dst_tz = {batch, 128, 112, 112};
    memory::dims pool1_kernel = {2, 2};
    memory::dims pool1_strides = {2, 2};
    memory::dims pool_padding = {0, 0};
        
    auto pool1_dst_md = memory::desc({pool1_dst_tz}, dt::f32, tag::any);

    // Create pooling primitive
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, conv2_dst_memory.get_desc(), pool1_dst_md,
            pool1_strides, pool1_kernel, pool_padding, pool_padding);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, eng);
    auto pool1_dst_memory = memory(pool1_pd.dst_desc(), eng);

    net.push_back(pooling_forward(pool1_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv2_dst_memory},
            {DNNL_ARG_DST, pool1_dst_memory}});

    // -----------------------------------------------------------
    // convolutional layer 3: 112x112x128
    memory::dims conv3_src_tz = {batch, 128, 112, 112};
    memory::dims conv3_weights_tz = {128, 64, 3, 3};
    memory::dims conv3_bias_tz = {128};
    memory::dims conv3_dst_tz = {batch, 128, 112, 112};
    memory::dims conv3_strides = {conv_stride, conv_stride};
    memory::dims conv3_padding = {padding, padding};

    std::vector<float> conv3_weights(product(conv3_weights_tz));
    std::vector<float> conv3_bias(product(conv3_bias_tz));

    // Create user memory
    auto conv3_user_weights_memory
            = memory({{conv3_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv3_weights.data(), conv3_user_weights_memory);
    auto conv3_user_bias_memory
            = memory({{conv3_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv3_bias.data(), conv3_user_bias_memory);

    // Create convolution memory descriptors with format_tag::any
    auto conv3_src_md = memory::desc({conv3_src_tz}, dt::f32, tag::any);
    auto conv3_weights_md = memory::desc({conv3_weights_tz}, dt::f32, tag::any);
    auto conv3_bias_md = memory::desc({conv3_bias_tz}, dt::f32, tag::any);
    auto conv3_dst_md = memory::desc({conv3_dst_tz}, dt::f32, tag::any);

    // Create convolution descriptor
    auto conv3_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv3_src_md, conv3_weights_md,
        conv3_bias_md,conv3_dst_md, conv3_strides, conv3_padding, conv3_padding);

    // Create convolution primitive descriptor 
    auto conv3_prim_desc = convolution_forward::primitive_desc(conv3_desc, eng);

    // Check if data and weights format required by convolution is different 
    // from the user format, if so reorder the memory layout
    auto conv3_src_memory = pool1_dst_memory;
    if (conv3_prim_desc.src_desc() != conv3_src_memory.get_desc()) {
        conv3_src_memory = memory(conv3_prim_desc.src_desc(), eng);
        net.push_back(reorder(pool1_dst_memory, conv3_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, pool1_dst_memory},
                {DNNL_ARG_TO, conv3_src_memory}});
    }

    auto conv3_weights_memory = conv3_user_weights_memory;
    if (conv3_prim_desc.weights_desc() != conv3_user_weights_memory.get_desc()) {
        conv3_weights_memory = memory(conv3_prim_desc.weights_desc(), eng);
        reorder(conv3_user_weights_memory, conv3_weights_memory)
                .execute(s, conv3_user_weights_memory, conv3_weights_memory);
    }

    // Create memory for output
    auto conv3_dst_memory = memory(conv3_prim_desc.dst_desc(),eng);

    // Create the convolution primitive
    net.push_back(convolution_forward(conv3_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv3_src_memory},
            {DNNL_ARG_WEIGHTS, conv3_weights_memory},
            {DNNL_ARG_BIAS, conv3_user_bias_memory},
            {DNNL_ARG_DST, conv3_dst_memory}});

    // -----------------------------------------------------------
    // ReLu3
    const float negative3_slope = 0.0f;

    // Create ReLu primitive
    auto relu3_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv3_dst_memory.get_desc(),
            negative3_slope);
    auto relu3_prim_desc = eltwise_forward::primitive_desc(relu3_desc, eng);

    net.push_back(eltwise_forward(relu3_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv3_dst_memory},
            {DNNL_ARG_DST, conv3_dst_memory}});

    // -----------------------------------------------------------
    // Convolutional layer 4: 112x112x128
    memory::dims conv4_src_tz = {batch, 128, 112, 112};
    memory::dims conv4_weights_tz = {128, 128, 3, 3}; 
    memory::dims conv4_bias_tz = {128};
    memory::dims conv4_dst_tz = {batch, 128, 112, 112};
    memory::dims conv4_strides = {conv_stride, conv_stride};
    memory::dims conv4_padding = {padding, padding};

    std::vector<float> conv4_weights(product(conv4_weights_tz));
    std::vector<float> conv4_bias(product(conv4_bias_tz));


    // Create user memory
    auto conv4_user_weights_memory
            = memory({{conv4_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv4_weights.data(), conv4_user_weights_memory);
    auto conv4_user_bias_memory
            = memory({{conv4_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv4_bias.data(), conv4_user_bias_memory);

    // Create convolution memory descriptors with format_tag::any
    auto conv4_src_md = memory::desc({conv4_src_tz}, dt::f32, tag::any);
    auto conv4_weights_md = memory::desc({conv4_weights_tz}, dt::f32, tag::any);
    auto conv4_bias_md = memory::desc({conv4_bias_tz}, dt::f32, tag::any);
    auto conv4_dst_md = memory::desc({conv4_dst_tz}, dt::f32, tag::any);

    // Create convolution descriptor
    auto conv4_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv4_src_md, conv4_weights_md,
        conv4_bias_md,conv4_dst_md, conv4_strides, conv4_padding, conv4_padding);

    // Create convolution primitive descriptor 
    auto conv4_prim_desc = convolution_forward::primitive_desc(conv4_desc, eng);

    // Check if data and weights format required by convolution is different 
    // from the user format, if so reorder the memory layout
    auto conv4_src_memory = conv3_dst_memory;
    if (conv4_prim_desc.src_desc() != conv4_src_memory.get_desc()) {
        conv4_src_memory = memory(conv4_prim_desc.src_desc(), eng);
        net.push_back(reorder(conv3_dst_memory, conv4_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv3_dst_memory},
                {DNNL_ARG_TO, conv4_src_memory}});
    }

    auto conv4_weights_memory = conv4_user_weights_memory;
    if (conv4_prim_desc.weights_desc() != conv4_user_weights_memory.get_desc()) {
        conv4_weights_memory = memory(conv4_prim_desc.weights_desc(), eng);
        reorder(conv4_user_weights_memory, conv4_weights_memory)
                .execute(s, conv4_user_weights_memory, conv4_weights_memory);
    }

    // Create memory for output
    auto conv4_dst_memory = memory(conv4_prim_desc.dst_desc(),eng);

    // Create the convolution primitive
    net.push_back(convolution_forward(conv4_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv4_src_memory},
            {DNNL_ARG_WEIGHTS, conv4_weights_memory},
            {DNNL_ARG_BIAS, conv4_user_bias_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});

    // -----------------------------------------------------------
    // ReLu4
    const float negative4_slope = 0.0f;

    // Create ReLu primitive
    auto relu4_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv4_dst_memory.get_desc(),
            negative4_slope);
    auto relu4_prim_desc = eltwise_forward::primitive_desc(relu4_desc, eng);

    net.push_back(eltwise_forward(relu4_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv4_dst_memory},
            {DNNL_ARG_DST, conv4_dst_memory}});

    // -----------------------------------------------------------
    // max pooling layer 2: 56x56
    // 112x112 -> 56x56
    memory::dims pool2_dst_tz = {batch, 256, 56, 56};
    memory::dims pool2_kernel = {2, 2};
    memory::dims pool2_strides = {2, 2};
    memory::dims pool_padding = {0, 0};
        
    auto pool2_dst_md = memory::desc({pool2_dst_tz}, dt::f32, tag::any);

    // Create pooling primitive
    auto pool2_desc = pooling_forward::desc(prop_kind::forward_inference,
            algorithm::pooling_max, conv4_dst_memory.get_desc(), pool2_dst_md,
            pool2_strides, pool2_kernel, pool_padding, pool_padding);
    auto pool2_pd = pooling_forward::primitive_desc(pool2_desc, eng);
    auto pool2_dst_memory = memory(pool2_pd.dst_desc(), eng);

    net.push_back(pooling_forward(pool2_pd));
    net_args.push_back({{DNNL_ARG_SRC, conv4_dst_memory},
            {DNNL_ARG_DST, pool2_dst_memory}});


    // -----------------------------------------------------------
    // convolutional layer 5: 56x56x256
    memory::dims conv5_src_tz = {batch, 256, 56, 56};
    memory::dims conv5_weights_tz = {256, 256, 3, 3};
    memory::dims conv5_bias_tz = {256};
    memory::dims conv5_dst_tz = {batch, 256, 56, 56};
    memory::dims conv5_strides = {conv_stride, conv_stride};
    memory::dims conv5_padding = {padding, padding};

    std::vector<float> conv5_weights(product(conv5_weights_tz));
    std::vector<float> conv5_bias(product(conv5_bias_tz));

    // Create user memory
    auto conv5_user_weights_memory
            = memory({{conv5_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv5_weights.data(), conv5_user_weights_memory);
    auto conv5_user_bias_memory
            = memory({{conv5_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv5_bias.data(), conv5_user_bias_memory);

    // Create convolution memory descriptors with format_tag::any
    auto conv5_src_md = memory::desc({conv5_src_tz}, dt::f32, tag::any);
    auto conv5_weights_md = memory::desc({conv5_weights_tz}, dt::f32, tag::any);
    auto conv5_bias_md = memory::desc({conv5_bias_tz}, dt::f32, tag::any);
    auto conv5_dst_md = memory::desc({conv5_dst_tz}, dt::f32, tag::any);

    // Create convolution descriptor
    auto conv5_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv5_src_md, conv5_weights_md,
        conv5_bias_md, conv5_dst_md, conv5_strides, conv5_padding, conv5_padding);

    // Create convolution primitive descriptor 
    auto conv5_prim_desc = convolution_forward::primitive_desc(conv5_desc, eng);

    // Check if data and weights format required by convolution is different 
    // from the user format, if so reorder the memory layout
    auto conv5_src_memory = pool2_dst_memory;
    if (conv5_prim_desc.src_desc() != conv5_src_memory.get_desc()) {
        conv5_src_memory = memory(conv5_prim_desc.src_desc(), eng);
        net.push_back(reorder(pool2_dst_memory, conv5_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, pool2_dst_memory},
                {DNNL_ARG_TO, conv5_src_memory}});
    }

    auto conv5_weights_memory = conv5_user_weights_memory;
    if (conv5_prim_desc.weights_desc() != conv5_user_weights_memory.get_desc()) {
        conv5_weights_memory = memory(conv5_prim_desc.weights_desc(), eng);
        reorder(conv5_user_weights_memory, conv5_weights_memory)
                .execute(s, conv5_user_weights_memory, conv5_weights_memory);
    }

    // Create memory for output
    auto conv5_dst_memory = memory(conv5_prim_desc.dst_desc(),eng);

    // Create the convolution primitive
    net.push_back(convolution_forward(conv5_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv5_src_memory},
            {DNNL_ARG_WEIGHTS, conv5_weights_memory},
            {DNNL_ARG_BIAS, conv5_user_bias_memory},
            {DNNL_ARG_DST, conv5_dst_memory}});

    // -----------------------------------------------------------
    // ReLu5
    const float negative5_slope = 0.0f;

    // Create ReLu primitive
    auto relu5_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv5_dst_memory.get_desc(),
            negative5_slope);
    auto relu5_prim_desc = eltwise_forward::primitive_desc(relu5_desc, eng);

    net.push_back(eltwise_forward(relu5_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv5_dst_memory},
            {DNNL_ARG_DST, conv5_dst_memory}});

    // -----------------------------------------------------------
    // convolutional layer 6: 56x56x256
    memory::dims conv6_src_tz = {batch, 256, 56, 56};
    memory::dims conv6_weights_tz = {256, 256, 3, 3};
    memory::dims conv6_bias_tz = {256};
    memory::dims conv6_dst_tz = {batch, 256, 56, 56};
    memory::dims conv6_strides = {conv_stride, conv_stride};
    memory::dims conv6_padding = {padding, padding};

    std::vector<float> conv6_weights(product(conv6_weights_tz));
    std::vector<float> conv6_bias(product(conv6_bias_tz));

    // Create user memory
    auto conv6_user_weights_memory
            = memory({{conv6_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv6_weights.data(), conv6_user_weights_memory);
    auto conv6_user_bias_memory
            = memory({{conv6_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv6_bias.data(), conv6_user_bias_memory);

    // Create convolution memory descriptors with format_tag::any
    auto conv6_src_md = memory::desc({conv6_src_tz}, dt::f32, tag::any);
    auto conv6_weights_md = memory::desc({conv6_weights_tz}, dt::f32, tag::any);
    auto conv6_bias_md = memory::desc({conv6_bias_tz}, dt::f32, tag::any);
    auto conv6_dst_md = memory::desc({conv6_dst_tz}, dt::f32, tag::any);

    // Create convolution descriptor
    auto conv6_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv6_src_md, conv6_weights_md,
        conv6_bias_md, conv6_dst_md, conv6_strides, conv6_padding, conv6_padding);

    // Create convolution primitive descriptor 
    auto conv6_prim_desc = convolution_forward::primitive_desc(conv6_desc, eng);

    // Check if data and weights format required by convolution is different 
    // from the user format, if so reorder the memory layout
    auto conv6_src_memory = conv5_dst_memory;
    if (conv6_prim_desc.src_desc() != conv6_src_memory.get_desc()) {
        conv6_src_memory = memory(conv6_prim_desc.src_desc(), eng);
        net.push_back(reorder(conv5_dst_memory, conv6_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv5_dst_memory},
                {DNNL_ARG_TO, conv6_src_memory}});
    }

    auto conv6_weights_memory = conv6_user_weights_memory;
    if (conv6_prim_desc.weights_desc() != conv6_user_weights_memory.get_desc()) {
        conv6_weights_memory = memory(conv6_prim_desc.weights_desc(), eng);
        reorder(conv6_user_weights_memory, conv6_weights_memory)
                .execute(s, conv6_user_weights_memory, conv6_weights_memory);
    }

    // Create memory for output
    auto conv6_dst_memory = memory(conv6_prim_desc.dst_desc(),eng);

    // Create the convolution primitive
    net.push_back(convolution_forward(conv6_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv6_src_memory},
            {DNNL_ARG_WEIGHTS, conv6_weights_memory},
            {DNNL_ARG_BIAS, conv6_user_bias_memory},
            {DNNL_ARG_DST, conv6_dst_memory}});

    // -----------------------------------------------------------
    // ReLu6
    const float negative6_slope = 0.0f;

    // Create ReLu primitive
    auto relu6_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv6_dst_memory.get_desc(),
            negative6_slope);
    auto relu6_prim_desc = eltwise_forward::primitive_desc(relu6_desc, eng);

    net.push_back(eltwise_forward(relu6_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv6_dst_memory},
            {DNNL_ARG_DST, conv6_dst_memory}});

    // -----------------------------------------------------------
    // convolutional layer 7: 56x56x256
    memory::dims conv7_src_tz = {batch, 256, 56, 56};
    memory::dims conv7_weights_tz = {256, 256, 3, 3};
    memory::dims conv7_bias_tz = {256};
    memory::dims conv7_dst_tz = {batch, 256, 56, 56};
    memory::dims conv7_strides = {conv_stride, conv_stride};
    memory::dims conv7_padding = {padding, padding};

    std::vector<float> conv7_weights(product(conv7_weights_tz));
    std::vector<float> conv7_bias(product(conv7_bias_tz));

    // Create user memory
    auto conv7_user_weights_memory
            = memory({{conv7_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv7_weights.data(), conv7_user_weights_memory);
    auto conv7_user_bias_memory
            = memory({{conv7_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv7_bias.data(), conv7_user_bias_memory);

    // Create convolution memory descriptors with format_tag::any
    auto conv7_src_md = memory::desc({conv7_src_tz}, dt::f32, tag::any);
    auto conv7_weights_md = memory::desc({conv7_weights_tz}, dt::f32, tag::any);
    auto conv7_bias_md = memory::desc({conv7_bias_tz}, dt::f32, tag::any);
    auto conv7_dst_md = memory::desc({conv7_dst_tz}, dt::f32, tag::any);

    // Create convolution descriptor
    auto conv7_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv7_src_md, conv7_weights_md,
        conv7_bias_md, conv7_dst_md, conv7_strides, conv7_padding, conv7_padding);

    // Create convolution primitive descriptor 
    auto conv7_prim_desc = convolution_forward::primitive_desc(conv7_desc, eng);

    // Check if data and weights format required by convolution is different 
    // from the user format, if so reorder the memory layout
    auto conv7_src_memory = conv6_dst_memory;
    if (conv7_prim_desc.src_desc() != conv7_src_memory.get_desc()) {
        conv7_src_memory = memory(conv7_prim_desc.src_desc(), eng);
        net.push_back(reorder(conv6_dst_memory, conv7_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, conv6_dst_memory},
                {DNNL_ARG_TO, conv7_src_memory}});
    }

    auto conv7_weights_memory = conv7_user_weights_memory;
    if (conv7_prim_desc.weights_desc() != conv7_user_weights_memory.get_desc()) {
        conv7_weights_memory = memory(conv7_prim_desc.weights_desc(), eng);
        reorder(conv7_user_weights_memory, conv7_weights_memory)
                .execute(s, conv7_user_weights_memory, conv7_weights_memory);
    }

    // Create memory for output
    auto conv7_dst_memory = memory(conv7_prim_desc.dst_desc(),eng);

    // Create the convolution primitive
    net.push_back(convolution_forward(conv7_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv7_src_memory},
            {DNNL_ARG_WEIGHTS, conv7_weights_memory},
            {DNNL_ARG_BIAS, conv7_user_bias_memory},
            {DNNL_ARG_DST, conv7_dst_memory}});

    // -----------------------------------------------------------
    // ReLu7
    const float negative7_slope = 0.0f;

    // Create ReLu primitive
    auto relu7_desc = eltwise_forward::desc(prop_kind::forward_inference,
            algorithm::eltwise_relu, conv7_dst_memory.get_desc(),
            negative7_slope);
    auto relu7_prim_desc = eltwise_forward::primitive_desc(relu7_desc, eng);

    net.push_back(eltwise_forward(relu7_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv7_dst_memory},
            {DNNL_ARG_DST, conv7_dst_memory}});

    // -----------------------------------------------------------
    // max pooling layer 3: 28x28x256
    

    // -----------------------------------------------------------
    // convolutional layer 8: 28x28x512

    // -----------------------------------------------------------
    // ReLu8

    // -----------------------------------------------------------
    // convolutional layer 9: 28x28x512

    // -----------------------------------------------------------
    // ReLu9

    // -----------------------------------------------------------
    // convolutional layer 10: 28x28x512

    // -----------------------------------------------------------
    // ReLu10

    // -----------------------------------------------------------
    // max pooling layer 4: 14x14x512

    // -----------------------------------------------------------
    // convolutional layer 11: 14x14x512

    // -----------------------------------------------------------
    // ReLu11

    // -----------------------------------------------------------
    // convolutional layer 12: 14x14x512

    // -----------------------------------------------------------
    // ReLu12

    // -----------------------------------------------------------
    // convolutional layer 13: 14x14x512

    // -----------------------------------------------------------
    // ReLu13

    // -----------------------------------------------------------
    // max pooling layer 5: 7x7x512

    // -----------------------------------------------------------
    // fully connected layer 1: 4096

    // -----------------------------------------------------------
    // ReLu

    // -----------------------------------------------------------
    // fully connected layer 2: 4096

    // -----------------------------------------------------------
    // ReLu

    // -----------------------------------------------------------
    // softmax layer: 1000





}