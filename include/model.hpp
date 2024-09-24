#ifndef MODEL_HPP
#define MODEL_HPP

#include "../include/tensor.hpp"
#include <memory>
#include <stdexcept>

// enum class for type safety
enum class LayerType {
    LINEAR = 1,
    RELU,
    SOFTMAX,
};

class Layer {
public:
    LayerType layer_type;
    std::unique_ptr<Tensor> weights;
    std::unique_ptr<Tensor> bias;
    std::unique_ptr<Tensor> input;  // For grad
    std::unique_ptr<Tensor> output; // For grad

    Layer(LayerType t, int input_size, int output_size) : layer_type(t) {
        if (t == LayerType::LINEAR) {
            weights = std::make_unique<Tensor>(std::vector<int>{input_size, output_size}, true, true);
            bias = std::make_unique<Tensor>(std::vector<int>{output_size}, true, true);
        }
    }

    Layer(const Layer&) = default;
    Layer(Layer&&) = default;
    ~Layer() = default;
};

class Model {
public:

    std::vector<std::unique_ptr<Layer>> layers;

    explicit Model(int num_layers) {
        layers.reserve(num_layers);
    }

    void add_layer(LayerType type, int input, int output) {
        if (layers.size() >= layers.capacity()) {
            throw std::runtime_error("Trying to add more layers than initially specified");
        }
        layers.push_back(std::make_unique<Layer>(type, input, output));
    }

};

// Function declarations
std::unique_ptr<Tensor> forward(Model& model, const Tensor& input);
void backward(Model& model, Tensor& pred, const Tensor& act);

#endif // MODEL_HPP
