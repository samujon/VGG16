#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
//#include "VGG16_helpers.hpp"
#include "/home/samjons/thesis/oneDNN/examples/example_utils.hpp"

using namespace dnnl;

// Helper function to create and initialize a convolution layer
std::tuple<primitive, std::unordered_map<int, memory>> create_convolution_layer(
    engine& eng, stream& s, memory& src, const memory::dims& src_tz, const memory::dims& weights_tz,
    const memory::dims& bias_tz, const memory::dims& dst_tz, const memory::dims& strides,
    const memory::dims& padding, float* weights_data, float* bias_data) {
    
    // Memory for weights and bias
    auto user_weights_memory = memory({{weights_tz}, memory::data_type::f32, memory::format_tag::oihw}, eng, weights_data);
    auto user_bias_memory = memory({{bias_tz}, memory::data_type::f32, memory::format_tag::x}, eng, bias_data);
    
    // Write data to memory
    write_to_dnnl_memory(weights_data, user_weights_memory);
    write_to_dnnl_memory(bias_data, user_bias_memory);

    // Descriptors
    auto src_md = memory::desc({src_tz}, memory::data_type::f32, memory::format_tag::any);
    auto weights_md = memory::desc({weights_tz}, memory::data_type::f32, memory::format_tag::any);
    auto bias_md = memory::desc({bias_tz}, memory::data_type::f32, memory::format_tag::any);
    auto dst_md = memory::desc({dst_tz}, memory::data_type::f32, memory::format_tag::any);
    
    // Convolution descriptor
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, src_md, weights_md, bias_md, dst_md,
        strides, padding, padding);
    
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, eng);
    
    // Reorder source memory if necessary
    auto conv_src = src;
    if (conv_pd.src_desc() != src.get_desc()) {
        conv_src = memory(conv_pd.src_desc(), eng);
        reorder(src, conv_src).execute(s, src, conv_src);
    }
    
    // Reorder weights memory if necessary
    auto conv_weights = user_weights_memory;
    if (conv_pd.weights_desc() != user_weights_memory.get_desc()) {
        conv_weights = memory(conv_pd.weights_desc(), eng);
        reorder(user_weights_memory, conv_weights).execute(s, user_weights_memory, conv_weights);
    }
    
    // Output memory
    auto conv_dst = memory(conv_pd.dst_desc(), eng);
    
    // Prepare arguments for the convolution primitive
    std::unordered_map<int, memory> conv_args = {
        {DNNL_ARG_SRC, conv_src},
        {DNNL_ARG_WEIGHTS, conv_weights},
        {DNNL_ARG_BIAS, user_bias_memory},
        {DNNL_ARG_DST, conv_dst}
    };
    
    return {convolution_forward(conv_pd), conv_args};
}

// Helper function to execute an activation layer (e.g., ReLU)
std::tuple<primitive, std::unordered_map<int, memory>> create_activation_layer(
    engine& eng, const memory& src, float slope) {
    
    // Activation descriptor
    auto relu_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, src.get_desc(), slope);
    auto relu_pd = eltwise_forward::primitive_desc(relu_desc, eng);
    
    // Arguments for the ReLU primitive
    std::unordered_map<int, memory> relu_args = {
        {DNNL_ARG_SRC, src},
        {DNNL_ARG_DST, src} // In-place operation
    };
    
    return {eltwise_forward(relu_pd), relu_args};
}
// CPU engine implementation

// In comparison with AlexNet, VGG16 does not LRN, local response normalization
// VGG16 D configuration
// A refactored version of VGG16.cpp
void VGG16(engine::kind engine_kind){
        std::cout << "Entered VGG16" << std::endl;
        using tag = memory::format_tag;
        using dt = memory::data_type;   
        // Create an engine
        engine eng(engine_kind, 0);
        stream s(eng);

        //Create network
        std::cout << "Create network" << std::endl;
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
        std::cout << "convolutional layer 1" << std::endl;
        memory::dims conv1_src_tz = {batch, input_channels, input_H, input_W};
        memory::dims conv1_weights_tz = {64, input_channels, 3, 3};
        memory::dims conv1_bias_tz = {64};
        memory::dims conv1_dst_tz = {batch, 64, input_H, input_W};
        memory::dims conv1_strides = {conv_stride, conv_stride};
        memory::dims conv1_padding = {padding, padding};

        // Allocate buffers for input data and weights, and create memory descriptors
        std::vector<float> user_src(batch * input_channels * input_H * input_W);
        //std::vector<float> user_dst(batch * 64 * input_H * input_W);
        std::vector<float> user_dst(batch*1000);
        std::vector<float> conv1_weights(product(conv1_weights_tz));
        std::vector<float> conv1_bias(product(conv1_bias_tz));

        // Create user memory
        auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
        write_to_dnnl_memory(user_src.data(), user_src_memory);
        
        // Create convolution layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv1 = create_convolution_layer(eng, s, user_src_memory, conv1_src_tz, 
                conv1_weights_tz, conv1_bias_tz, conv1_dst_tz, conv1_strides, conv1_padding, conv1_weights.data(), conv1_bias.data());
        auto conv1_dst_memory = std::get<1>(conv1).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv1));
        net_args.push_back(std::get<1>(conv1));

        // -----------------------------------------------------------
        // ReLu1
        std::cout << "ReLu1" << std::endl;
        const float negative1_slope = 0.0f;
        
        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu1 = create_activation_layer(eng, conv1_dst_memory, negative1_slope);
        net.push_back(std::get<0>(relu1));
        net_args.push_back(std::get<1>(relu1));


        // -----------------------------------------------------------
        // convolutional layer 2: 224x224x64
        std::cout << "convolutional layer 2" << std::endl;
        memory::dims conv2_src_tz = {batch, 64, 224, 224};
        memory::dims conv2_weights_tz = {64, 64, 3, 3};
        memory::dims conv2_bias_tz = {64};
        memory::dims conv2_dst_tz = {batch, 64, 224, 224};
        memory::dims conv2_strides = {conv_stride, conv_stride};
        memory::dims conv2_padding = {padding, padding};
        
        // Allocate buffers for input data and weights, and create memory descriptors
        std::vector<float> conv2_weights(product(conv2_weights_tz));
        std::vector<float> conv2_bias(product(conv2_bias_tz));

        // Create convolution layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv2 = create_convolution_layer(eng, s, conv1_dst_memory, conv2_src_tz, 
                conv2_weights_tz, conv2_bias_tz, conv2_dst_tz, conv2_strides, conv2_padding, conv2_weights.data(), conv2_bias.data());
        auto conv2_dst_memory = std::get<1>(conv2).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv2));
        net_args.push_back(std::get<1>(conv2));

        // -----------------------------------------------------------
        // ReLu2
        std::cout << "ReLu2" << std::endl;
        const float negative2_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu2 = create_activation_layer(eng, conv2_dst_memory, negative2_slope);
        net.push_back(std::get<0>(relu2));
        net_args.push_back(std::get<1>(relu2));

        // -----------------------------------------------------------
        // max pooling layer 1: 112x112
        // 224x224 -> 112x112
        std::cout << "max pooling layer 1" << std::endl;
        memory::dims pool1_dst_tz = {batch, 64, 112, 112};
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
        std::cout << "convolutional layer 3" << std::endl;
        memory::dims conv3_src_tz = {batch, 64, 112, 112};
        memory::dims conv3_weights_tz = {128, 64, 3, 3};
        memory::dims conv3_bias_tz = {128};
        memory::dims conv3_dst_tz = {batch, 128, 112, 112};
        memory::dims conv3_strides = {conv_stride, conv_stride};
        memory::dims conv3_padding = {padding, padding};

        std::vector<float> conv3_weights(product(conv3_weights_tz));
        std::vector<float> conv3_bias(product(conv3_bias_tz));

        // Create convolution layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv3 = create_convolution_layer(eng, s, pool1_dst_memory, conv3_src_tz, 
                conv3_weights_tz, conv3_bias_tz, conv3_dst_tz, conv3_strides, conv3_padding, conv3_weights.data(), conv3_bias.data());
        auto conv3_dst_memory = std::get<1>(conv3).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv3));
        net_args.push_back(std::get<1>(conv3));
        
        // -----------------------------------------------------------
        // ReLu3
        std::cout << "ReLu3" << std::endl;
        const float negative3_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu3 = create_activation_layer(eng, conv3_dst_memory, negative3_slope);
        net.push_back(std::get<0>(relu3));
        net_args.push_back(std::get<1>(relu3));

        // -----------------------------------------------------------
        // Convolutional layer 4: 112x112x128
        std::cout << "convolutional layer 4" << std::endl;
        memory::dims conv4_src_tz = {batch, 128, 112, 112};
        memory::dims conv4_weights_tz = {128, 128, 3, 3}; 
        memory::dims conv4_bias_tz = {128};
        memory::dims conv4_dst_tz = {batch, 128, 112, 112};
        memory::dims conv4_strides = {conv_stride, conv_stride};
        memory::dims conv4_padding = {padding, padding};

        std::vector<float> conv4_weights(product(conv4_weights_tz));
        std::vector<float> conv4_bias(product(conv4_bias_tz));

        // Create convolutiion layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv4 = create_convolution_layer(eng, s, conv3_dst_memory, conv4_src_tz, 
                conv4_weights_tz, conv4_bias_tz, conv4_dst_tz, conv4_strides, conv4_padding, conv4_weights.data(), conv4_bias.data());
        auto conv4_dst_memory = std::get<1>(conv4).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv4));
        net_args.push_back(std::get<1>(conv4));

        // -----------------------------------------------------------
        // ReLu4
        std::cout << "ReLu4" << std::endl;
        const float negative4_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu4 = create_activation_layer(eng, conv4_dst_memory, negative4_slope);
        net.push_back(std::get<0>(relu4));
        net_args.push_back(std::get<1>(relu4));

        // -----------------------------------------------------------
        // max pooling layer 2: 56x56
        // 112x112 -> 56x56
        std::cout << "max pooling layer 2" << std::endl;
        memory::dims pool2_dst_tz = {batch, 128, 56, 56};
        memory::dims pool2_kernel = {2, 2};
        memory::dims pool2_strides = {2, 2};

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
        std::cout << "convolutional layer 5" << std::endl;
        memory::dims conv5_src_tz = {batch, 128, 56, 56};
        memory::dims conv5_weights_tz = {256, 128, 3, 3};
        memory::dims conv5_bias_tz = {256};
        memory::dims conv5_dst_tz = {batch, 256, 56, 56};
        memory::dims conv5_strides = {conv_stride, conv_stride};
        memory::dims conv5_padding = {padding, padding};

        std::vector<float> conv5_weights(product(conv5_weights_tz));
        std::vector<float> conv5_bias(product(conv5_bias_tz));

        // Create convolution layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv5 = create_convolution_layer(eng, s, pool2_dst_memory, conv5_src_tz, 
                conv5_weights_tz, conv5_bias_tz, conv5_dst_tz, conv5_strides, conv5_padding, conv5_weights.data(), conv5_bias.data());
        auto conv5_dst_memory = std::get<1>(conv5).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv5));
        net_args.push_back(std::get<1>(conv5));
        
        // -----------------------------------------------------------
        // ReLu5
        std::cout << "ReLu5" << std::endl;
        const float negative5_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu5 = create_activation_layer(eng, conv5_dst_memory, negative5_slope);
        net.push_back(std::get<0>(relu5));
        net_args.push_back(std::get<1>(relu5));

        // -----------------------------------------------------------
        // convolutional layer 6: 56x56x256
        std::cout << "convolutional layer 6" << std::endl;
        memory::dims conv6_src_tz = {batch, 256, 56, 56};
        memory::dims conv6_weights_tz = {256, 256, 3, 3};
        memory::dims conv6_bias_tz = {256};
        memory::dims conv6_dst_tz = {batch, 256, 56, 56};
        memory::dims conv6_strides = {conv_stride, conv_stride};
        memory::dims conv6_padding = {padding, padding};

        std::vector<float> conv6_weights(product(conv6_weights_tz));
        std::vector<float> conv6_bias(product(conv6_bias_tz));

        // Create convolution layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv6 = create_convolution_layer(eng, s, conv5_dst_memory, conv6_src_tz, 
                conv6_weights_tz, conv6_bias_tz, conv6_dst_tz, conv6_strides, conv6_padding, conv6_weights.data(), conv6_bias.data());
        auto conv6_dst_memory = std::get<1>(conv6).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv6));
        net_args.push_back(std::get<1>(conv6));

        // -----------------------------------------------------------
        // ReLu6
        std::cout << "ReLu6" << std::endl;
        const float negative6_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu6 = create_activation_layer(eng, conv6_dst_memory, negative6_slope);
        net.push_back(std::get<0>(relu6));
        net_args.push_back(std::get<1>(relu6));

        // -----------------------------------------------------------
        // convolutional layer 7: 56x56x256
        std::cout << "convolutional layer 7" << std::endl;
        memory::dims conv7_src_tz = {batch, 256, 56, 56};
        memory::dims conv7_weights_tz = {256, 256, 3, 3};
        memory::dims conv7_bias_tz = {256};
        memory::dims conv7_dst_tz = {batch, 256, 56, 56};
        memory::dims conv7_strides = {conv_stride, conv_stride};
        memory::dims conv7_padding = {padding, padding};

        std::vector<float> conv7_weights(product(conv7_weights_tz));
        std::vector<float> conv7_bias(product(conv7_bias_tz));

        // Create convolution layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv7 = create_convolution_layer(eng, s, conv6_dst_memory, conv7_src_tz, 
                conv7_weights_tz, conv7_bias_tz, conv7_dst_tz, conv7_strides, conv7_padding, conv7_weights.data(), conv7_bias.data());
        auto conv7_dst_memory = std::get<1>(conv7).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv7));
        net_args.push_back(std::get<1>(conv7));

        // -----------------------------------------------------------
        // ReLu7
        std::cout << "ReLu7" << std::endl;
        const float negative7_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu7 = create_activation_layer(eng, conv7_dst_memory, negative7_slope);
        net.push_back(std::get<0>(relu7));
        net_args.push_back(std::get<1>(relu7));

        // -----------------------------------------------------------
        // max pooling layer 3: 28x28x256
        // 56x56 -> 28x28
        std::cout << "max pooling layer 3" << std::endl;
        memory::dims pool3_dst_tz = {batch, 256, 28, 28};
        memory::dims pool3_kernel = {2, 2};
        memory::dims pool3_strides = {2, 2};

        auto pool3_dst_md = memory::desc({pool3_dst_tz}, dt::f32, tag::any);

        // Create pooling primitive
        auto pool3_desc = pooling_forward::desc(prop_kind::forward_inference,
        algorithm::pooling_max, conv7_dst_memory.get_desc(), pool3_dst_md,
        pool3_strides, pool3_kernel, pool_padding, pool_padding);
        auto pool3_pd = pooling_forward::primitive_desc(pool3_desc, eng);
        auto pool3_dst_memory = memory(pool3_pd.dst_desc(), eng);

        net.push_back(pooling_forward(pool3_pd));
        net_args.push_back({{DNNL_ARG_SRC, conv7_dst_memory},
        {DNNL_ARG_DST, pool3_dst_memory}});

        // -----------------------------------------------------------
        // convolutional layer 8: 28x28x512
        std::cout << "convolutional layer 8" << std::endl;
        memory::dims conv8_src_tz = {batch, 256, 28, 28};
        memory::dims conv8_weights_tz = {512, 256, 3, 3};
        memory::dims conv8_bias_tz = {512};
        memory::dims conv8_dst_tz = {batch, 512, 28, 28};
        memory::dims conv8_strides = {conv_stride, conv_stride};
        memory::dims conv8_padding = {padding, padding};

        std::vector<float> conv8_weights(product(conv8_weights_tz));
        std::vector<float> conv8_bias(product(conv8_bias_tz));

        // Create convolutional layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv8 = create_convolution_layer(eng, s, pool3_dst_memory, conv8_src_tz, 
                conv8_weights_tz, conv8_bias_tz, conv8_dst_tz, conv8_strides, conv8_padding, conv8_weights.data(), conv8_bias.data());
        auto conv8_dst_memory = std::get<1>(conv8).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv8));
        net_args.push_back(std::get<1>(conv8));

        // -----------------------------------------------------------
        // ReLu8
        std::cout << "ReLu8" << std::endl;
        const float negative8_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu8 = create_activation_layer(eng, conv8_dst_memory, negative8_slope);
        net.push_back(std::get<0>(relu8));
        net_args.push_back(std::get<1>(relu8));

        // -----------------------------------------------------------
        // convolutional layer 9: 28x28x512
        std::cout << "convolutional layer 9" << std::endl;
        memory::dims conv9_src_tz = {batch, 512, 28, 28};
        memory::dims conv9_weights_tz = {512, 512, 3, 3};
        memory::dims conv9_bias_tz = {512};
        memory::dims conv9_dst_tz = {batch, 512, 28, 28};
        memory::dims conv9_strides = {conv_stride, conv_stride};
        memory::dims conv9_padding = {padding, padding};

        std::vector<float> conv9_weights(product(conv9_weights_tz));
        std::vector<float> conv9_bias(product(conv9_bias_tz));

        // Create convolutional layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv9 = create_convolution_layer(eng, s, conv8_dst_memory, conv9_src_tz, 
                conv9_weights_tz, conv9_bias_tz, conv9_dst_tz, conv9_strides, conv9_padding, conv9_weights.data(), conv9_bias.data());
        auto conv9_dst_memory = std::get<1>(conv9).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv9));
        net_args.push_back(std::get<1>(conv9));

        // -----------------------------------------------------------
        // ReLu9
        std::cout << "ReLu9" << std::endl;
        const float negative9_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu9 = create_activation_layer(eng, conv9_dst_memory, negative9_slope);
        net.push_back(std::get<0>(relu9));
        net_args.push_back(std::get<1>(relu9));

        // -----------------------------------------------------------
        // convolutional layer 10: 28x28x512
        std::cout << "convolutional layer 10" << std::endl;
        memory::dims conv10_src_tz = {batch, 512, 28, 28};
        memory::dims conv10_weights_tz = {512, 512, 3, 3};
        memory::dims conv10_bias_tz = {512};
        memory::dims conv10_dst_tz = {batch, 512, 28, 28};
        memory::dims conv10_strides = {conv_stride, conv_stride};
        memory::dims conv10_padding = {padding, padding};

        std::vector<float> conv10_weights(product(conv10_weights_tz));
        std::vector<float> conv10_bias(product(conv10_bias_tz));

        // Create convolutional layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv10 = create_convolution_layer(eng, s, conv9_dst_memory, conv10_src_tz, 
                conv10_weights_tz, conv10_bias_tz, conv10_dst_tz, conv10_strides, conv10_padding, conv10_weights.data(), conv10_bias.data());
        auto conv10_dst_memory = std::get<1>(conv10).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv10));
        net_args.push_back(std::get<1>(conv10));

        // -----------------------------------------------------------
        // ReLu10
        std::cout << "ReLu10" << std::endl;
        const float negative10_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu10 = create_activation_layer(eng, conv10_dst_memory, negative10_slope);
        net.push_back(std::get<0>(relu10));
        net_args.push_back(std::get<1>(relu10));

        // -----------------------------------------------------------
        // max pooling layer 4: 14x14x512
        // 28x28 -> 14x14
        std::cout << "max pooling layer 4" << std::endl;
        memory::dims pool4_dst_tz = {batch, 512, 14, 14};
        memory::dims pool4_kernel = {2, 2};
        memory::dims pool4_strides = {2, 2};

        auto pool4_dst_md = memory::desc({pool4_dst_tz}, dt::f32, tag::any);

        // Create pooling primitive
        auto pool4_desc = pooling_forward::desc(prop_kind::forward_inference,
        algorithm::pooling_max, conv10_dst_memory.get_desc(), pool4_dst_md,
        pool4_strides, pool4_kernel, pool_padding, pool_padding);
        auto pool4_pd = pooling_forward::primitive_desc(pool4_desc, eng);
        auto pool4_dst_memory = memory(pool4_pd.dst_desc(), eng);

        net.push_back(pooling_forward(pool4_pd));
        net_args.push_back({{DNNL_ARG_SRC, conv10_dst_memory},
        {DNNL_ARG_DST, pool4_dst_memory}});

        // -----------------------------------------------------------
        // convolutional layer 11: 14x14x512
        std::cout << "convolutional layer 11" << std::endl;
        memory::dims conv11_src_tz = {batch, 512, 14, 14};
        memory::dims conv11_weights_tz = {512, 512, 3, 3};
        memory::dims conv11_bias_tz = {512};
        memory::dims conv11_dst_tz = {batch, 512, 14, 14};
        memory::dims conv11_strides = {conv_stride, conv_stride};
        memory::dims conv11_padding = {padding, padding};

        std::vector<float> conv11_weights(product(conv11_weights_tz));
        std::vector<float> conv11_bias(product(conv11_bias_tz));

        // Create convolutional layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv11 = create_convolution_layer(eng, s, pool4_dst_memory, conv11_src_tz, 
                conv11_weights_tz, conv11_bias_tz, conv11_dst_tz, conv11_strides, conv11_padding, conv11_weights.data(), conv11_bias.data());
        auto conv11_dst_memory = std::get<1>(conv11).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv11));
        net_args.push_back(std::get<1>(conv11));

        // -----------------------------------------------------------
        // ReLu11
        std::cout << "ReLu11" << std::endl;
        const float negative11_slope = 0.0f;

        // Create ReLu primitive
        auto relu11_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv11_dst_memory.get_desc(),
        negative11_slope);
        auto relu11_prim_desc = eltwise_forward::primitive_desc(relu11_desc, eng);

        net.push_back(eltwise_forward(relu11_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv11_dst_memory},
        {DNNL_ARG_DST, conv11_dst_memory}});

        // -----------------------------------------------------------
        // convolutional layer 12: 14x14x512
        std::cout << "convolutional layer 12" << std::endl;
        memory::dims conv12_src_tz = {batch, 512, 14, 14};
        memory::dims conv12_weights_tz = {512, 512, 3, 3};
        memory::dims conv12_bias_tz = {512};
        memory::dims conv12_dst_tz = {batch, 512, 14, 14};
        memory::dims conv12_strides = {conv_stride, conv_stride};
        memory::dims conv12_padding = {padding, padding};

        std::vector<float> conv12_weights(product(conv12_weights_tz));
        std::vector<float> conv12_bias(product(conv12_bias_tz));

        // Create convolutional layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv12 = create_convolution_layer(eng, s, conv11_dst_memory, conv12_src_tz, 
                conv12_weights_tz, conv12_bias_tz, conv12_dst_tz, conv12_strides, conv12_padding, conv12_weights.data(), conv12_bias.data());
        auto conv12_dst_memory = std::get<1>(conv12).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv12));
        net_args.push_back(std::get<1>(conv12));

        // -----------------------------------------------------------
        // ReLu12
        std::cout << "ReLu12" << std::endl;
        const float negative12_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu12 = create_activation_layer(eng, conv12_dst_memory, negative12_slope);
        net.push_back(std::get<0>(relu12));
        net_args.push_back(std::get<1>(relu12));

        // -----------------------------------------------------------
        // convolutional layer 13: 14x14x512
        std::cout << "convolutional layer 13" << std::endl;
        memory::dims conv13_src_tz = {batch, 512, 14, 14};
        memory::dims conv13_weights_tz = {512, 512, 3, 3};
        memory::dims conv13_bias_tz = {512};
        memory::dims conv13_dst_tz = {batch, 512, 14, 14};
        memory::dims conv13_strides = {conv_stride, conv_stride};
        memory::dims conv13_padding = {padding, padding};

        std::vector<float> conv13_weights(product(conv13_weights_tz));
        std::vector<float> conv13_bias(product(conv13_bias_tz));

        // Create convolutional layer
        std::tuple<primitive, std::unordered_map<int, memory>> conv13 = create_convolution_layer(eng, s, conv12_dst_memory, conv13_src_tz, 
                conv13_weights_tz, conv13_bias_tz, conv13_dst_tz, conv13_strides, conv13_padding, conv13_weights.data(), conv13_bias.data());
        auto conv13_dst_memory = std::get<1>(conv13).at(DNNL_ARG_DST);
        net.push_back(std::get<0>(conv13));
        net_args.push_back(std::get<1>(conv13));
        
        // -----------------------------------------------------------
        // ReLu13
        std::cout << "ReLu13" << std::endl;
        const float negative13_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu13 = create_activation_layer(eng, conv13_dst_memory, negative13_slope);
        net.push_back(std::get<0>(relu13));
        net_args.push_back(std::get<1>(relu13));

        // -----------------------------------------------------------
        // max pooling layer 5: 7x7x512
        // 14x14 -> 7x7
        std::cout << "max pooling layer 5" << std::endl;
        memory::dims pool5_dst_tz = {batch, 512, 7, 7};
        memory::dims pool5_kernel = {2, 2};
        memory::dims pool5_strides = {2, 2};

        auto pool5_dst_md = memory::desc({pool5_dst_tz}, dt::f32, tag::any);

        // Create pooling primitive
        auto pool5_desc = pooling_forward::desc(prop_kind::forward_inference,
        algorithm::pooling_max, conv13_dst_memory.get_desc(), pool5_dst_md,
        pool5_strides, pool5_kernel, pool_padding, pool_padding);
        auto pool5_pd = pooling_forward::primitive_desc(pool5_desc, eng);
        auto pool5_dst_memory = memory(pool5_pd.dst_desc(), eng);

        net.push_back(pooling_forward(pool5_pd));
        net_args.push_back({{DNNL_ARG_SRC, conv13_dst_memory},
        {DNNL_ARG_DST, pool5_dst_memory}});

        // -----------------------------------------------------------
        // fully connected layer 1: 4096
        std::cout << "fully connected layer 1" << std::endl;
        memory::dims fc1_src_tz = {batch, 512, 7, 7};
        memory::dims fc1_weights_tz = {4096, 512, 7, 7};
        memory::dims fc1_bias_tz = {4096};
        memory::dims fc1_dst_tz = {batch, 4096};

        std::vector<float> fc1_weights(product(fc1_weights_tz));
        std::vector<float> fc1_bias(product(fc1_bias_tz));
        
        // Create user memory
        auto fc1_user_weights_memory = memory({{fc1_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(fc1_weights.data(), fc1_user_weights_memory);
        auto fc1_user_bias_memory = memory({{fc1_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(fc1_bias.data(), fc1_user_bias_memory);

        // Create memory descriptors for convolution data
        auto fc1_src_md = memory::desc({fc1_src_tz}, dt::f32, tag::any);
        auto fc1_bias_md = memory::desc({fc1_bias_tz}, dt::f32, tag::any);
        auto fc1_weights_md = memory::desc({fc1_weights_tz}, dt::f32, tag::any);
        auto fc1_dst_md = memory::desc{{fc1_dst_tz}, dt::f32, tag::any};

        // Create inner product (fully connected) descriptor
        auto fc1_desc = inner_product_forward::desc(prop_kind::forward_inference,
            fc1_src_md, fc1_weights_md, fc1_bias_md, fc1_dst_md);

        auto fc1_prim_desc = inner_product_forward::primitive_desc(fc1_desc, eng);

        // Check if reorder needed 
        auto fc1_src_memory = pool5_dst_memory;
        if (fc1_prim_desc.src_desc() != pool5_dst_memory.get_desc()) {
        fc1_src_memory = memory(fc1_prim_desc.src_desc(), eng);
        net.push_back(reorder(pool5_dst_memory, fc1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, pool5_dst_memory},
        {DNNL_ARG_TO, fc1_src_memory}});
        }

        // Create memory for output
        auto fc1_dst_memory = memory(fc1_prim_desc.dst_desc(), eng);

        // Add FC layer to the network
        net.push_back(inner_product_forward(fc1_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc1_src_memory},
        {DNNL_ARG_WEIGHTS, fc1_user_weights_memory},
        {DNNL_ARG_BIAS, fc1_user_bias_memory},
        {DNNL_ARG_DST, fc1_dst_memory}});

        // -----------------------------------------------------------
        // ReLu14
        std::cout << "ReLu14" << std::endl;
        const float negative14_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu14 = create_activation_layer(eng, fc1_dst_memory, negative14_slope);
        net.push_back(std::get<0>(relu14));
        net_args.push_back(std::get<1>(relu14));

        // -----------------------------------------------------------
        // fully connected layer 2: 4096
        std::cout << "fully connected layer 2" << std::endl;
        memory::dims fc2_src_tz = {batch, 4096};
        memory::dims fc2_weights_tz = {4096, 4096};
        memory::dims fc2_bias_tz = {4096};
        memory::dims fc2_dst_tz = {batch, 4096};

        std::vector<float> fc2_weights(product(fc2_weights_tz));
        std::vector<float> fc2_bias(product(fc2_bias_tz));

        // Create user memory
        auto fc2_user_weights_memory = memory({{fc2_weights_tz}, dt::f32, tag::nc}, eng);
        write_to_dnnl_memory(fc2_weights.data(), fc2_user_weights_memory);
        auto fc2_user_bias_memory = memory({{fc2_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(fc2_bias.data(), fc2_user_bias_memory);

        // Create memory descriptors for convolution data
        auto fc2_src_md = memory::desc({fc2_src_tz}, dt::f32, tag::any);
        auto fc2_bias_md = memory::desc({fc2_bias_tz}, dt::f32, tag::any);
        auto fc2_weights_md = memory::desc({fc2_weights_tz}, dt::f32, tag::any);
        auto fc2_dst_md = memory::desc{{fc2_dst_tz}, dt::f32, tag::any};

        // Create inner product (fully connected) descriptor
        auto fc2_desc = inner_product_forward::desc(prop_kind::forward_inference,
        fc2_src_md, fc2_weights_md, fc2_bias_md, fc2_dst_md);
        auto fc2_prim_desc = inner_product_forward::primitive_desc(fc2_desc, eng);

        // Check if reorder needed 
        auto fc2_src_memory = fc1_dst_memory;
        if (fc2_prim_desc.src_desc() != fc1_dst_memory.get_desc()) {
        fc2_src_memory = memory(fc2_prim_desc.src_desc(), eng);
        net.push_back(reorder(fc1_dst_memory, fc2_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, fc1_dst_memory},
        {DNNL_ARG_TO, fc2_src_memory}});
        }

        // Create memory for output
        auto fc2_dst_memory = memory(fc2_prim_desc.dst_desc(), eng);

        // Add FC layer to the network
        net.push_back(inner_product_forward(fc2_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc2_src_memory},
        {DNNL_ARG_WEIGHTS, fc2_user_weights_memory},
        {DNNL_ARG_BIAS, fc2_user_bias_memory},
        {DNNL_ARG_DST, fc2_dst_memory}});

        // -----------------------------------------------------------
        // ReLu15
        std::cout << "ReLu15" << std::endl;
        const float negative15_slope = 0.0f;

        // Create ReLu primitive
        auto relu15_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, fc2_dst_memory.get_desc(),
        negative15_slope);
        auto relu15_prim_desc = eltwise_forward::primitive_desc(relu15_desc, eng);

        net.push_back(eltwise_forward(relu15_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc2_dst_memory},
        {DNNL_ARG_DST, fc2_dst_memory}});

        // -----------------------------------------------------------
        // fully connected layer 3: 1000
        std::cout << "fully connected layer 3" << std::endl;
        memory::dims fc3_src_tz = {batch, 4096};
        memory::dims fc3_weights_tz = {1000, 4096};
        memory::dims fc3_bias_tz = {1000};
        memory::dims fc3_dst_tz = {batch, 1000};

        std::vector<float> fc3_weights(product(fc3_weights_tz));
        std::vector<float> fc3_bias(product(fc3_bias_tz));

        // Create user memory
        auto fc3_user_weights_memory = memory({{fc3_weights_tz}, dt::f32, tag::nc}, eng);
        write_to_dnnl_memory(fc3_weights.data(), fc3_user_weights_memory);
        auto fc3_user_bias_memory = memory({{fc3_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(fc3_bias.data(), fc3_user_bias_memory);
        //auto user_dst_memory = memory({{fc3_dst_tz}, dt::f32, tag::nc}, eng);
        //write_to_dnnl_memory(user_dst.data(), user_dst_memory);

        // Create memory descriptors for convolution data
        auto fc3_src_md = memory::desc({fc3_src_tz}, dt::f32, tag::any);
        auto fc3_bias_md = memory::desc({fc3_bias_tz}, dt::f32, tag::any);
        auto fc3_weights_md = memory::desc({fc3_weights_tz}, dt::f32, tag::any);
        auto fc3_dst_md = memory::desc{{fc3_dst_tz}, dt::f32, tag::any};

        // Create inner product (fully connected) descriptor
        auto fc3_desc = inner_product_forward::desc(prop_kind::forward_inference,
        fc3_src_md, fc3_weights_md, fc3_bias_md, fc3_dst_md);
        auto fc3_prim_desc = inner_product_forward::primitive_desc(fc3_desc, eng);

        // Check if reorder needed 
        auto fc3_src_memory = fc2_dst_memory;
        if (fc3_prim_desc.src_desc() != fc2_dst_memory.get_desc()) {
        fc3_src_memory = memory(fc3_prim_desc.src_desc(), eng);
        net.push_back(reorder(fc2_dst_memory, fc3_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, fc2_dst_memory},
        {DNNL_ARG_TO, fc3_src_memory}});
        }

        // Create memory for output
        auto fc3_dst_memory = memory(fc3_prim_desc.dst_desc(), eng);

        // Add FC layer to the network
        net.push_back(inner_product_forward(fc3_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc3_src_memory},
        {DNNL_ARG_WEIGHTS, fc3_user_weights_memory},
        {DNNL_ARG_BIAS, fc3_user_bias_memory},
        {DNNL_ARG_DST, fc3_dst_memory}});

        // -----------------------------------------------------------
        // ReLu16
        std::cout << "ReLu16" << std::endl;
        const float negative16_slope = 0.0f;

        // Create ReLu primitive
        std::tuple<primitive, std::unordered_map<int, memory>> relu16 = create_activation_layer(eng, fc3_dst_memory, negative16_slope);
        net.push_back(std::get<0>(relu16));
        net_args.push_back(std::get<1>(relu16));
        
        // -----------------------------------------------------------
        // Softmax
        std::cout << "Softmax" << std::endl;
        auto softmax_desc = softmax_forward::desc(prop_kind::forward_inference,
        fc3_dst_memory.get_desc(), 1);
        auto softmax_prim_desc = softmax_forward::primitive_desc(softmax_desc, eng);
        auto softmax_dst_memory = memory(softmax_prim_desc.dst_desc(), eng);
        softmax_forward softmax_prim(softmax_prim_desc);

        // Execute softmax
        softmax_prim.execute(s, {{DNNL_ARG_SRC, fc3_dst_memory},
        {DNNL_ARG_DST, softmax_dst_memory}});

        // -----------------------------------------------------------
        // Execute model
        std::cout << "Execute model" << std::endl;
        net.at(0).execute(s, net_args.at(0));
        s.wait();

}

int main(int argc, char **argv) {
        auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        //std::cout << "Start time: " << begin << std::endl;
        VGG16(parse_engine_kind(argc, argv));
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        std::cout << "VGG16 executed" << std::endl;
        //std::cout << "End time: " << end << std::endl;
        std::cout << "Total time:" << (end - begin)/1000.0 << "s" << std::endl;
}