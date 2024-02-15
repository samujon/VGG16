#include "VGG16_helpers.hpp"

namespace dnnl {
namespace VGG16_helpers {

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

} // namespace vgg16_helpers
} // namespace dnnl
