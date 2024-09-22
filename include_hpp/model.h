#ifndef MODEL_HPP
#define MODEL_HPP

#include "tensor.hpp"
#include <memory>

// enum class for type safety
enum class LAYER_TYPE {
    LINEAR_LAYER = 1,
    RELU_LAYER,
    SOFTMAX_LAYER,
};

class Layer {
public:
    LAYER_TYPE layer_type;
    std::unique_ptr<Tensor> weights;
    std::unique_ptr<Tensor> bias;
    std::unique_ptr<Tensor> input;  // For grad
    std::unique_ptr<Tensor> output; // For grad

    Layer() = default;
    ~Layer() = default;
};

class Model {
public:
    std::vector<Layer> layers;
    int layer_size;
    int layer_index;

    Model(int num_layers) : layer_size(num_layers), layer_index(0) {
        layers.reserve(num_layers);
    }

    void add_layer(LAYER_TYPE type, int input, int output);
    std::unique_ptr<Tensor> forward(const Tensor& input);
    void backwards(const Tensor& pred, const Tensor& act);

    ~Model() = default;
};

std::unique_ptr<Model> create_model(int num_layers);

#endif // MODEL_HPP
