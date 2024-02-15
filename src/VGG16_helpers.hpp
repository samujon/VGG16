#ifndef VGG16_HELPERS_H
#define VGG16_HELPERS_H

#include <vector>
#include <unordered_map>
#include <dnnl.hpp>
#include "/home/samjons/thesis/oneDNN/examples/example_utils.hpp"

// Declare namespace to organize the code, optional
namespace dnnl{
namespace vgg16_helpers {

std::tuple<primitive, std::unordered_map<int, memory>> create_convolution_layer(
    engine& eng, stream& s, memory& src, const memory::dims& src_tz, const memory::dims& weights_tz,
    const memory::dims& bias_tz, const memory::dims& dst_tz, const memory::dims& strides,
    const memory::dims& padding, float* weights_data, float* bias_data) {}

std::tuple<primitive, std::unordered_map<int, memory>> create_activation_layer(
    engine& eng, const memory& src, float slope) {}

} // namespace vgg16_helpers
} // namespace dnnl
#endif // DNNL_HELPERS_H
