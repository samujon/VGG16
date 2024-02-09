#ifndef VGG16_HPP
#define VGG16_HPP

#include <iostream>
#include <vector>

class VGG16 {
public:
    VGG16();
    ~VGG16();

    void loadModel(const std::string& modelPath);
    std::vector<float> predict(const std::vector<float>& input);

private:
    // Add private member variables and helper functions here

};

#endif // VGG16_HPP
