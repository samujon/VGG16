#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#include "/home/samjons/thesis/oneDNN/examples/example_utils.hpp"

using namespace dnnl;

// CPU engine implementation

// In comparison with AlexNet, VGG16 does not LRN, local response normalization
// VGG16 D configuration
// A procedural version of VGG16
void VGG16(engine::kind engine_kind){
        auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
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
        std::vector<float> user_dst(batch*1000);
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
        reorder(user_src_memory, conv1_src_memory)
                .execute(s, user_src_memory, conv1_src_memory);
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
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu1
        std::cout << "ReLu1" << std::endl;
        const float negative1_slope = 0.0f;

        // Create ReLu primitive
        auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv1_dst_memory.get_desc(),
        negative1_slope);
        auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);

        net.push_back(eltwise_forward(relu1_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv1_dst_memory},
        {DNNL_ARG_DST, conv1_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        conv2_bias_md, conv2_dst_md, conv2_strides, conv2_padding, conv2_padding);
        
        // Create convolution primitive descriptor 
        auto conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, eng);
        // Check if data and weights format required by convolution is different 
        // from the user format, if so reorder the memory layout
        auto conv2_src_memory = conv1_dst_memory;
        if (conv2_prim_desc.src_desc() != conv2_src_memory.get_desc()) {
        conv2_src_memory = memory(conv2_prim_desc.src_desc(), eng);
        reorder(conv1_dst_memory, conv2_src_memory)
                .execute(s, conv1_dst_memory, conv2_src_memory);
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
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu2
        std::cout << "ReLu2" << std::endl;
        const float negative2_slope = 0.0f;

        // Create ReLu primitive
        auto relu2_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv2_dst_memory.get_desc(),
        negative2_slope);
        auto relu2_prim_desc = eltwise_forward::primitive_desc(relu2_desc, eng);

        net.push_back(eltwise_forward(relu2_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv2_dst_memory},
        {DNNL_ARG_DST, conv2_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        std::cout << "Layer: "<< net.size() << std::endl;

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
        reorder(pool1_dst_memory, conv3_src_memory)
                .execute(s, pool1_dst_memory, conv3_src_memory);
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
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu3
        std::cout << "ReLu3" << std::endl;
        const float negative3_slope = 0.0f;

        // Create ReLu primitive
        auto relu3_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv3_dst_memory.get_desc(),
        negative3_slope);
        auto relu3_prim_desc = eltwise_forward::primitive_desc(relu3_desc, eng);

        net.push_back(eltwise_forward(relu3_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv3_dst_memory},
        {DNNL_ARG_DST, conv3_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        reorder(conv3_dst_memory, conv3_src_memory)
                .execute(s, conv3_dst_memory, conv3_src_memory);

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
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu4
        std::cout << "ReLu4" << std::endl;
        const float negative4_slope = 0.0f;

        // Create ReLu primitive
        auto relu4_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv4_dst_memory.get_desc(),
        negative4_slope);
        auto relu4_prim_desc = eltwise_forward::primitive_desc(relu4_desc, eng);

        net.push_back(eltwise_forward(relu4_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv4_dst_memory},
        {DNNL_ARG_DST, conv4_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        std::cout << "Layer: "<< net.size() << std::endl;

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
        reorder(pool2_dst_memory, conv5_src_memory)
                .execute(s, pool2_dst_memory, conv5_src_memory);
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
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu5
        std::cout << "ReLu5" << std::endl;
        const float negative5_slope = 0.0f;

        // Create ReLu primitive
        auto relu5_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv5_dst_memory.get_desc(),
        negative5_slope);
        auto relu5_prim_desc = eltwise_forward::primitive_desc(relu5_desc, eng);

        net.push_back(eltwise_forward(relu5_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv5_dst_memory},
        {DNNL_ARG_DST, conv5_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        reorder(conv5_dst_memory, conv6_src_memory)
                .execute(s, conv5_dst_memory, conv6_src_memory);
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
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu6
        std::cout << "ReLu6" << std::endl;
        const float negative6_slope = 0.0f;

        // Create ReLu primitive
        auto relu6_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv6_dst_memory.get_desc(),
        negative6_slope);
        auto relu6_prim_desc = eltwise_forward::primitive_desc(relu6_desc, eng);

        net.push_back(eltwise_forward(relu6_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv6_dst_memory},
        {DNNL_ARG_DST, conv6_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        reorder(conv6_dst_memory, conv7_src_memory)
                .execute(s, conv6_dst_memory, conv7_src_memory);
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
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu7
        std::cout << "ReLu7" << std::endl;
        const float negative7_slope = 0.0f;

        // Create ReLu primitive
        auto relu7_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv7_dst_memory.get_desc(),
        negative7_slope);
        auto relu7_prim_desc = eltwise_forward::primitive_desc(relu7_desc, eng);

        net.push_back(eltwise_forward(relu7_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv7_dst_memory},
        {DNNL_ARG_DST, conv7_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        std::cout << "Layer: "<< net.size() << std::endl;

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

        // Create user memory
        auto conv8_user_weights_memory
        = memory({{conv8_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv8_weights.data(), conv8_user_weights_memory);
        auto conv8_user_bias_memory
        = memory({{conv8_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv8_bias.data(), conv8_user_bias_memory);

        // Create convolution memory descriptors with format_tag::any
        auto conv8_src_md = memory::desc({conv8_src_tz}, dt::f32, tag::any);
        auto conv8_weights_md = memory::desc({conv8_weights_tz}, dt::f32, tag::any);
        auto conv8_bias_md = memory::desc({conv8_bias_tz}, dt::f32, tag::any);
        auto conv8_dst_md = memory::desc({conv8_dst_tz}, dt::f32, tag::any);

        // Create convolution descriptor
        auto conv8_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv8_src_md, conv8_weights_md,
        conv8_bias_md, conv8_dst_md, conv8_strides, conv8_padding, conv8_padding);

        // Create convolution primitive descriptor 
        auto conv8_prim_desc = convolution_forward::primitive_desc(conv8_desc, eng);

        // Check if data and weights format required by convolution is different 
        // from the user format, if so reorder the memory layout
        auto conv8_src_memory = pool3_dst_memory;
        if (conv8_prim_desc.src_desc() != conv8_src_memory.get_desc()) {
        conv8_src_memory = memory(conv8_prim_desc.src_desc(), eng);
        reorder(pool3_dst_memory, conv8_src_memory)
                .execute(s, pool3_dst_memory, conv8_src_memory);
        }

        auto conv8_weights_memory = conv8_user_weights_memory;
        if (conv8_prim_desc.weights_desc() != conv8_user_weights_memory.get_desc()) {
        conv8_weights_memory = memory(conv8_prim_desc.weights_desc(), eng);
        reorder(conv8_user_weights_memory, conv8_weights_memory)
                .execute(s, conv8_user_weights_memory, conv8_weights_memory);
        }

        // Create memory for output
        auto conv8_dst_memory = memory(conv8_prim_desc.dst_desc(),eng);

        // Create the convolution primitive
        net.push_back(convolution_forward(conv8_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv8_src_memory},
        {DNNL_ARG_WEIGHTS, conv8_weights_memory},
        {DNNL_ARG_BIAS, conv8_user_bias_memory},
        {DNNL_ARG_DST, conv8_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu8
        std::cout << "ReLu8" << std::endl;
        const float negative8_slope = 0.0f;

        // Create ReLu primitive
        auto relu8_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv8_dst_memory.get_desc(),
        negative8_slope);
        auto relu8_prim_desc = eltwise_forward::primitive_desc(relu8_desc, eng);

        net.push_back(eltwise_forward(relu8_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv8_dst_memory},
        {DNNL_ARG_DST, conv8_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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

        // Create user memory
        auto conv9_user_weights_memory
        = memory({{conv9_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv9_weights.data(), conv9_user_weights_memory);
        auto conv9_user_bias_memory
        = memory({{conv9_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv9_bias.data(), conv9_user_bias_memory);

        // Create convolution memory descriptors with format_tag::any
        auto conv9_src_md = memory::desc({conv9_src_tz}, dt::f32, tag::any);
        auto conv9_weights_md = memory::desc({conv9_weights_tz}, dt::f32, tag::any);
        auto conv9_bias_md = memory::desc({conv9_bias_tz}, dt::f32, tag::any);
        auto conv9_dst_md = memory::desc({conv9_dst_tz}, dt::f32, tag::any);

        // Create convolution descriptor
        auto conv9_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv9_src_md, conv9_weights_md,
        conv9_bias_md, conv9_dst_md, conv9_strides, conv9_padding, conv9_padding);

        // Create convolution primitive descriptor 
        auto conv9_prim_desc = convolution_forward::primitive_desc(conv9_desc, eng);

        // Check if data and weights format required by convolution is different 
        // from the user format, if so reorder the memory layout
        auto conv9_src_memory = conv8_dst_memory;
        if (conv9_prim_desc.src_desc() != conv9_src_memory.get_desc()) {
        conv9_src_memory = memory(conv9_prim_desc.src_desc(), eng);
        reorder(conv8_dst_memory, conv9_src_memory)
                .execute(s, conv8_dst_memory, conv9_src_memory);
        }

        auto conv9_weights_memory = conv9_user_weights_memory;
        if (conv9_prim_desc.weights_desc() != conv9_user_weights_memory.get_desc()) {
        conv9_weights_memory = memory(conv9_prim_desc.weights_desc(), eng);
        reorder(conv9_user_weights_memory, conv9_weights_memory)
                .execute(s, conv9_user_weights_memory, conv9_weights_memory);
        }

        // Create memory for output
        auto conv9_dst_memory = memory(conv9_prim_desc.dst_desc(),eng);

        // Create the convolution primitive
        net.push_back(convolution_forward(conv9_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv9_src_memory},
        {DNNL_ARG_WEIGHTS, conv9_weights_memory},
        {DNNL_ARG_BIAS, conv9_user_bias_memory},
        {DNNL_ARG_DST, conv9_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu9
        std::cout << "ReLu9" << std::endl;
        const float negative9_slope = 0.0f;

        // Create ReLu primitive
        auto relu9_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv9_dst_memory.get_desc(),
        negative9_slope);
        auto relu9_prim_desc = eltwise_forward::primitive_desc(relu9_desc, eng);

        net.push_back(eltwise_forward(relu9_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv9_dst_memory},
        {DNNL_ARG_DST, conv9_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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

        // Create user memory
        auto conv10_user_weights_memory
        = memory({{conv10_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv10_weights.data(), conv10_user_weights_memory);
        auto conv10_user_bias_memory
        = memory({{conv10_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv10_bias.data(), conv10_user_bias_memory);

        // Create convolution memory descriptors with format_tag::any
        auto conv10_src_md = memory::desc({conv10_src_tz}, dt::f32, tag::any);
        auto conv10_weights_md = memory::desc({conv10_weights_tz}, dt::f32, tag::any);
        auto conv10_bias_md = memory::desc({conv10_bias_tz}, dt::f32, tag::any);
        auto conv10_dst_md = memory::desc({conv10_dst_tz}, dt::f32, tag::any);

        // Create convolution descriptor
        auto conv10_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv10_src_md, conv10_weights_md,
        conv10_bias_md, conv10_dst_md, conv10_strides, conv10_padding, conv10_padding);

        // Create convolution primitive descriptor 
        auto conv10_prim_desc = convolution_forward::primitive_desc(conv10_desc, eng);

        // Check if data and weights format required by convolution is different 
        // from the user format, if so reorder the memory layout
        auto conv10_src_memory = conv9_dst_memory;
        if (conv10_prim_desc.src_desc() != conv10_src_memory.get_desc()) {
        conv10_src_memory = memory(conv10_prim_desc.src_desc(), eng);
        reorder(conv9_dst_memory, conv10_src_memory)
                .execute(s, conv9_dst_memory, conv10_src_memory);
        }

        auto conv10_weights_memory = conv10_user_weights_memory;
        if (conv10_prim_desc.weights_desc() != conv10_user_weights_memory.get_desc()) {
        conv10_weights_memory = memory(conv10_prim_desc.weights_desc(), eng);
        reorder(conv10_user_weights_memory, conv10_weights_memory)
                .execute(s, conv10_user_weights_memory, conv10_weights_memory);
        }

        // Create memory for output
        auto conv10_dst_memory = memory(conv10_prim_desc.dst_desc(),eng);

        // Create the convolution primitive
        net.push_back(convolution_forward(conv10_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv10_src_memory},
        {DNNL_ARG_WEIGHTS, conv10_weights_memory},
        {DNNL_ARG_BIAS, conv10_user_bias_memory},
        {DNNL_ARG_DST, conv10_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu10
        std::cout << "ReLu10" << std::endl;
        const float negative10_slope = 0.0f;

        // Create ReLu primitive
        auto relu10_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv10_dst_memory.get_desc(),
        negative10_slope);
        auto relu10_prim_desc = eltwise_forward::primitive_desc(relu10_desc, eng);

        net.push_back(eltwise_forward(relu10_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv10_dst_memory},
        {DNNL_ARG_DST, conv10_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        std::cout << "Layer: "<< net.size() << std::endl;

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

        // Create user memory
        auto conv11_user_weights_memory
        = memory({{conv11_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv11_weights.data(), conv11_user_weights_memory);
        auto conv11_user_bias_memory
        = memory({{conv11_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv11_bias.data(), conv11_user_bias_memory);

        // Create convolution memory descriptors with format_tag::any
        auto conv11_src_md = memory::desc({conv11_src_tz}, dt::f32, tag::any);
        auto conv11_weights_md = memory::desc({conv11_weights_tz}, dt::f32, tag::any);
        auto conv11_bias_md = memory::desc({conv11_bias_tz}, dt::f32, tag::any);
        auto conv11_dst_md = memory::desc({conv11_dst_tz}, dt::f32, tag::any);

        // Create convolution descriptor
        auto conv11_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv11_src_md, conv11_weights_md,
        conv11_bias_md, conv11_dst_md, conv11_strides, conv11_padding, conv11_padding);

        // Create convolution primitive descriptor 
        auto conv11_prim_desc = convolution_forward::primitive_desc(conv11_desc, eng);

        // Check if data and weights format required by convolution is different 
        // from the user format, if so reorder the memory layout
        auto conv11_src_memory = pool4_dst_memory;
        if (conv11_prim_desc.src_desc() != conv11_src_memory.get_desc()) {
        conv11_src_memory = memory(conv11_prim_desc.src_desc(), eng);
        reorder(pool4_dst_memory, conv11_src_memory)
                .execute(s, pool4_dst_memory, conv11_src_memory);
        }

        auto conv11_weights_memory = conv11_user_weights_memory;
        if (conv11_prim_desc.weights_desc() != conv11_user_weights_memory.get_desc()) {
        conv11_weights_memory = memory(conv11_prim_desc.weights_desc(), eng);
        reorder(conv11_user_weights_memory, conv11_weights_memory)
                .execute(s, conv11_user_weights_memory, conv11_weights_memory);
        }

        // Create memory for output
        auto conv11_dst_memory = memory(conv11_prim_desc.dst_desc(),eng);

        // Create the convolution primitive
        net.push_back(convolution_forward(conv11_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv11_src_memory},
        {DNNL_ARG_WEIGHTS, conv11_weights_memory},
        {DNNL_ARG_BIAS, conv11_user_bias_memory},
        {DNNL_ARG_DST, conv11_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        std::cout << "Layer: "<< net.size() << std::endl;

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

        // Create user memory
        auto conv12_user_weights_memory
        = memory({{conv12_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv12_weights.data(), conv12_user_weights_memory);
        auto conv12_user_bias_memory
        = memory({{conv12_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv12_bias.data(), conv12_user_bias_memory);

        // Create convolution memory descriptors with format_tag::any
        auto conv12_src_md = memory::desc({conv12_src_tz}, dt::f32, tag::any);
        auto conv12_weights_md = memory::desc({conv12_weights_tz}, dt::f32, tag::any);
        auto conv12_bias_md = memory::desc({conv12_bias_tz}, dt::f32, tag::any);
        auto conv12_dst_md = memory::desc({conv12_dst_tz}, dt::f32, tag::any);

        // Create convolution descriptor
        auto conv12_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv12_src_md, conv12_weights_md,
        conv12_bias_md, conv12_dst_md, conv12_strides, conv12_padding, conv12_padding);

        // Create convolution primitive descriptor 
        auto conv12_prim_desc = convolution_forward::primitive_desc(conv12_desc, eng);

        // Check if data and weights format required by convolution is different 
        // from the user format, if so reorder the memory layout
        auto conv12_src_memory = conv11_dst_memory;
        if (conv12_prim_desc.src_desc() != conv12_src_memory.get_desc()) {
        conv12_src_memory = memory(conv12_prim_desc.src_desc(), eng);
        reorder(conv11_dst_memory, conv12_src_memory)
                .execute(s, conv11_dst_memory, conv12_src_memory);
        }

        auto conv12_weights_memory = conv12_user_weights_memory;
        if (conv12_prim_desc.weights_desc() != conv12_user_weights_memory.get_desc()) {
        conv12_weights_memory = memory(conv12_prim_desc.weights_desc(), eng);
        reorder(conv12_user_weights_memory, conv12_weights_memory)
                .execute(s, conv12_user_weights_memory, conv12_weights_memory);
        }

        // Create memory for output
        auto conv12_dst_memory = memory(conv12_prim_desc.dst_desc(),eng);

        // Create the convolution primitive
        net.push_back(convolution_forward(conv12_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv12_src_memory},
        {DNNL_ARG_WEIGHTS, conv12_weights_memory},
        {DNNL_ARG_BIAS, conv12_user_bias_memory},
        {DNNL_ARG_DST, conv12_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu12
        std::cout << "ReLu12" << std::endl;
        const float negative12_slope = 0.0f;

        // Create ReLu primitive
        auto relu12_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv12_dst_memory.get_desc(),
        negative12_slope);
        auto relu12_prim_desc = eltwise_forward::primitive_desc(relu12_desc, eng);

        net.push_back(eltwise_forward(relu12_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv12_dst_memory},
        {DNNL_ARG_DST, conv12_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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

        // Create user memory
        auto conv13_user_weights_memory
        = memory({{conv13_weights_tz}, dt::f32, tag::oihw}, eng);
        write_to_dnnl_memory(conv13_weights.data(), conv13_user_weights_memory);
        auto conv13_user_bias_memory
        = memory({{conv13_bias_tz}, dt::f32, tag::x}, eng);
        write_to_dnnl_memory(conv13_bias.data(), conv13_user_bias_memory);

        // Create convolution memory descriptors with format_tag::any
        auto conv13_src_md = memory::desc({conv13_src_tz}, dt::f32, tag::any);
        auto conv13_weights_md = memory::desc({conv13_weights_tz}, dt::f32, tag::any);
        auto conv13_bias_md = memory::desc({conv13_bias_tz}, dt::f32, tag::any);
        auto conv13_dst_md = memory::desc({conv13_dst_tz}, dt::f32, tag::any);

        // Create convolution descriptor
        auto conv13_desc = convolution_forward::desc(prop_kind::forward_inference,
        algorithm::convolution_direct, conv13_src_md, conv13_weights_md,
        conv13_bias_md, conv13_dst_md, conv13_strides, conv13_padding, conv13_padding);

        // Create convolution primitive descriptor 
        auto conv13_prim_desc = convolution_forward::primitive_desc(conv13_desc, eng);

        // Check if data and weights format required by convolution is different 
        // from the user format, if so reorder the memory layout
        auto conv13_src_memory = conv12_dst_memory;
        if (conv13_prim_desc.src_desc() != conv13_src_memory.get_desc()) {
        conv13_src_memory = memory(conv13_prim_desc.src_desc(), eng);
        reorder(conv12_dst_memory, conv13_src_memory)
                .execute(s, conv12_dst_memory, conv13_src_memory);
        }

        auto conv13_weights_memory = conv13_user_weights_memory;
        if (conv13_prim_desc.weights_desc() != conv13_user_weights_memory.get_desc()) {
        conv13_weights_memory = memory(conv13_prim_desc.weights_desc(), eng);
        reorder(conv13_user_weights_memory, conv13_weights_memory)
                .execute(s, conv13_user_weights_memory, conv13_weights_memory);
        }

        // Create memory for output
        auto conv13_dst_memory = memory(conv13_prim_desc.dst_desc(),eng);

        // Create the convolution primitive
        net.push_back(convolution_forward(conv13_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv13_src_memory},
        {DNNL_ARG_WEIGHTS, conv13_weights_memory},
        {DNNL_ARG_BIAS, conv13_user_bias_memory},
        {DNNL_ARG_DST, conv13_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu13
        std::cout << "ReLu13" << std::endl;
        const float negative13_slope = 0.0f;

        // Create ReLu primitive
        auto relu13_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, conv13_dst_memory.get_desc(),
        negative13_slope);
        auto relu13_prim_desc = eltwise_forward::primitive_desc(relu13_desc, eng);

        net.push_back(eltwise_forward(relu13_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, conv13_dst_memory},
        {DNNL_ARG_DST, conv13_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        std::cout << "Layer: "<< net.size() << std::endl;

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
        auto fc1_desc = inner_product_forward::desc(prop_kind::forward,
            fc1_src_md, fc1_weights_md, fc1_bias_md, fc1_dst_md);

        auto fc1_prim_desc = inner_product_forward::primitive_desc(fc1_desc, eng);

        // Check if reorder needed 
        auto fc1_src_memory = pool5_dst_memory;
        if (fc1_prim_desc.src_desc() != pool5_dst_memory.get_desc()) {
        fc1_src_memory = memory(fc1_prim_desc.src_desc(), eng);
        net.push_back(reorder(pool5_dst_memory, fc1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, pool5_dst_memory},{
        DNNL_ARG_TO, fc1_src_memory}});
        }

        // Check if reorder needed 
        auto fc1_weights_memory = fc1_user_weights_memory;
        if (fc1_prim_desc.weights_desc() != fc1_user_weights_memory.get_desc()) {
        fc1_weights_memory = memory(fc1_prim_desc.weights_desc(), eng);
        reorder(fc1_user_weights_memory, fc1_weights_memory)
        .execute(s, fc1_user_weights_memory, fc1_weights_memory);
        }

        // Create memory for output
        auto fc1_dst_memory = memory(fc1_prim_desc.dst_desc(), eng);

        // Add FC layer to the network
        net.push_back(inner_product_forward(fc1_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc1_src_memory},
        {DNNL_ARG_WEIGHTS, fc1_weights_memory},
        {DNNL_ARG_BIAS, fc1_user_bias_memory},
        {DNNL_ARG_DST, fc1_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu14
        std::cout << "ReLu14" << std::endl;
        const float negative14_slope = 0.0f;

        // Create ReLu primitive
        auto relu14_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, fc1_dst_memory.get_desc(),
        negative14_slope);
        auto relu14_prim_desc = eltwise_forward::primitive_desc(relu14_desc, eng);

        net.push_back(eltwise_forward(relu14_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc1_dst_memory},
        {DNNL_ARG_DST, fc1_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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
        auto fc2_desc = inner_product_forward::desc(prop_kind::forward,
        fc2_src_md, fc2_weights_md, fc2_bias_md, fc2_dst_md);
        auto fc2_prim_desc = inner_product_forward::primitive_desc(fc2_desc, eng);

        // Check if reorder needed 
        auto fc2_src_memory = fc1_dst_memory;
        if (fc2_prim_desc.src_desc() != fc1_dst_memory.get_desc()) {
        fc2_src_memory = memory(fc2_prim_desc.src_desc(), eng);
        reorder(fc1_dst_memory, fc2_src_memory)
        .execute(s, fc1_dst_memory, fc2_src_memory);
        }

        // Check if reorder needed
        auto fc2_weights_memory = fc2_user_weights_memory;
        if (fc2_prim_desc.weights_desc() != fc2_user_weights_memory.get_desc()) {
        fc2_weights_memory = memory(fc2_prim_desc.weights_desc(), eng);
        reorder(fc2_user_weights_memory, fc2_weights_memory)
        .execute(s, fc2_user_weights_memory, fc2_weights_memory);
        }
        
        // Create memory for output
        auto fc2_dst_memory = memory(fc2_prim_desc.dst_desc(), eng);

        // Add FC layer to the network
        net.push_back(inner_product_forward(fc2_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc2_src_memory},
        {DNNL_ARG_WEIGHTS, fc2_weights_memory},
        {DNNL_ARG_BIAS, fc2_user_bias_memory},
        {DNNL_ARG_DST, fc2_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu15
        std::cout << "ReLu15" << std::endl;
        const float negative15_slope = 0.0f;

        // Create ReLu primitive
        auto relu15_desc = eltwise_forward::desc(prop_kind::forward,
        algorithm::eltwise_relu, fc2_dst_memory.get_desc(),
        negative15_slope);
        auto relu15_prim_desc = eltwise_forward::primitive_desc(relu15_desc, eng);

        net.push_back(eltwise_forward(relu15_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc2_dst_memory},
        {DNNL_ARG_DST, fc2_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

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

        // Create memory descriptors for convolution data
        auto fc3_src_md = memory::desc({fc3_src_tz}, dt::f32, tag::any);
        auto fc3_bias_md = memory::desc({fc3_bias_tz}, dt::f32, tag::any);
        auto fc3_weights_md = memory::desc({fc3_weights_tz}, dt::f32, tag::any);
        auto fc3_dst_md = memory::desc{{fc3_dst_tz}, dt::f32, tag::any};

        // Create inner product (fully connected) descriptor
        auto fc3_desc = inner_product_forward::desc(prop_kind::forward,
        fc3_src_md, fc3_weights_md, fc3_bias_md, fc3_dst_md);
        auto fc3_prim_desc = inner_product_forward::primitive_desc(fc3_desc, eng);

        // Check if reorder needed 
        auto fc3_src_memory = fc2_dst_memory;
        if (fc3_prim_desc.src_desc() != fc2_dst_memory.get_desc()) {
        fc3_src_memory = memory(fc3_prim_desc.src_desc(), eng);
        reorder(fc2_dst_memory, fc3_src_memory)
        .execute(s, fc2_dst_memory, fc3_src_memory);
        }

        // Check if reorder needed
        auto fc3_weights_memory = fc3_user_weights_memory;
        if (fc3_prim_desc.weights_desc() != fc3_user_weights_memory.get_desc()) {
        fc3_weights_memory = memory(fc3_prim_desc.weights_desc(), eng);
        reorder(fc3_user_weights_memory, fc3_weights_memory)
        .execute(s, fc3_user_weights_memory, fc3_weights_memory);
        }

        // Create memory for output
        auto fc3_dst_memory = memory(fc3_prim_desc.dst_desc(), eng);

        // Add FC layer to the network
        net.push_back(inner_product_forward(fc3_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc3_src_memory},
        {DNNL_ARG_WEIGHTS, fc3_weights_memory},
        {DNNL_ARG_BIAS, fc3_user_bias_memory},
        {DNNL_ARG_DST, fc3_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // ReLu16
        std::cout << "ReLu16" << std::endl;
        const float negative16_slope = 0.0f;

        // Create ReLu primitive
        auto relu16_desc = eltwise_forward::desc(prop_kind::forward_inference,
        algorithm::eltwise_relu, fc3_dst_memory.get_desc(),
        negative16_slope);
        auto relu16_prim_desc = eltwise_forward::primitive_desc(relu15_desc, eng);

        net.push_back(eltwise_forward(relu16_prim_desc));
        net_args.push_back({{DNNL_ARG_SRC, fc3_dst_memory},
        {DNNL_ARG_DST, fc3_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;

        // -----------------------------------------------------------
        // Softmax
        std::cout << "Softmax" << std::endl;
        auto softmax_desc = softmax_forward::desc(prop_kind::forward,
        fc3_dst_memory.get_desc(), 1);
        auto softmax_prim_desc = softmax_forward::primitive_desc(softmax_desc, eng);
        auto user_dst_memory = memory(softmax_prim_desc.dst_desc(), eng);
        softmax_forward softmax_prim(softmax_prim_desc);
        //auto user_dst_memory = memory({{batch, 1000}, dt::f32, tag::nc}, eng);
        //auto user_dst_memory = memory({{fc3_dst_tz}, dt::f32, tag::nc}, eng);
        write_to_dnnl_memory(user_dst.data(), user_dst_memory);

        // Execute softmax
        softmax_prim.execute(s, {{DNNL_ARG_SRC, fc3_dst_memory},
        {DNNL_ARG_DST, user_dst_memory}});
        if(user_dst_memory != fc3_dst_memory){
        reorder(user_dst_memory, fc3_dst_memory)
        .execute(s, user_dst_memory, fc3_dst_memory);
        }

        /*if (softmax_dst_memory != user_dst_memory) {
        reorder(softmax_dst_memory, user_dst_memory)
        .execute(s, softmax_dst_memory, user_dst_memory);
        }*/

        net.push_back(softmax_prim_desc);
        net_args.push_back({{DNNL_ARG_SRC, fc3_dst_memory},
        {DNNL_ARG_DST, user_dst_memory}});
        std::cout << "Layer: "<< net.size() << std::endl;
        auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count();
        std::cout << "Time to create network:" << (end - begin)/1000.0 << "s" << std::endl;

        // -----------------------------------------------------------
        // Execute model
        std::cout << "Execute model" << std::endl;
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i){
            //std::cout << "Executing layer " << i + 1 << std::endl;
            net.at(i).execute(s, net_args.at(i));
            std::cout << "Executed layer " << i + 1 << std::endl;
        }
        s.wait();
        std::vector<float> result(1000);
        read_from_dnnl_memory(result.data(), user_dst_memory);
        
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