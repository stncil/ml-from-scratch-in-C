#include "../include/tensor.hpp"
#include "../include/model.hpp"

#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>

std::unique_ptr<Tensor> forward(Model& model, const Tensor& input) {
    const Tensor* x = &input;
    // for (auto i: x->data)
    // std::cout << x->data.size() << ' ';
    // std::cout << "!!!";
    std::unique_ptr<Tensor> next;

    for (auto& layer : model.layers) {
        layer->input = std::make_unique<Tensor>(x->shape, true);
        layer->input->data = x->data;

        switch (layer->layer_type) {
            case LayerType::LINEAR: {
                const int batch_size = x->shape[0];
                const int input_size = x->shape[1];
                const int output_size = layer->bias->shape[0];
                next = std::make_unique<Tensor>(std::vector<int>{batch_size, output_size}, true);
                
                for (int b = 0; b < batch_size; ++b) {
                    for (int j = 0; j < output_size; ++j) {
                        double sum = 0.0;
                        for (int i = 0; i < input_size; ++i) {
                            sum += x->data[b * input_size + i] * layer->weights->data[i * output_size + j];
                        }
                        next->data[b * output_size + j] = sum + layer->bias->data[j];
                    }
                }
                break;
            }
            case LayerType::RELU: {
                next = std::make_unique<Tensor>(x->shape, x->require_grad);
                std::transform(x->data.begin(), x->data.end(), next->data.begin(),
                               [](double val) { return std::max(0.0, val); });
                break;
            }
            case LayerType::SOFTMAX: {
                const int batch_size = x->shape[0];
                const int class_count = x->shape[1];
                next = std::make_unique<Tensor>(x->shape, x->require_grad);
                
                for (int b = 0; b < batch_size; ++b) {
                    const double max_val = *std::max_element(x->data.begin() + b * class_count, 
                                                             x->data.begin() + ((b + 1) * (class_count)-1));
                    std::vector<double> exps(class_count);
                    double sum_of_exps = 0.0;
                    
                    for (int j = 0; j < class_count; ++j) {
                        exps[j] = std::exp(x->data[b * class_count + j] - max_val);
                        sum_of_exps += exps[j];
                    }
                    
                    for (int j = 0; j < class_count; ++j) {
                        next->data[b * class_count + j] = exps[j] / sum_of_exps;
                    }
                }
                break;
            }
        }

        layer->output = std::move(next);
        x = layer->output.get();
    }

    return std::make_unique<Tensor>(*x);
}
// Backward function would be implemented similarly, with efficient operations
// Helper function for element-wise multiplication
std::vector<double> element_wise_multiply(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::multiplies<>());
    return result;
}

// Helper function for matrix multiplication
std::vector<double> matmul(const std::vector<double>& a, const std::vector<double>& b, int m, int n, int p) {
    std::vector<double> result(m * p, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i * p + j] += a[i * n + k] * b[k * p + j];
            }
        }
    }
    return result;
}

// Helper function for transposing a matrix
std::vector<double> transpose(const std::vector<double>& matrix, int rows, int cols) {
    std::vector<double> result(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
    return result;
}

void backward(Model& model, Tensor& pred, const Tensor& actual) {
    int last_layer = model.layers.size() - 1;
    const int batch_size = pred.shape[0];

    // Compute initial gradient (assuming cross-entropy loss with softmax output)
    std::vector<double> grad = pred.grad;
    for (size_t i = 0; i < actual.data.size(); ++i) {
        double targ = (pred.data[i] == actual.data[i]) ? 1.0 : 0.0;
        pred.grad[i] += targ;
    }

    for (int i = last_layer; i >= 0; --i) {
        Layer& layer = *model.layers[i];

        switch (layer.layer_type) {
            case LayerType::SOFTMAX:
                // Softmax gradient is already computed in the initial step
                break;

            case LayerType::LINEAR: {
                const int input_size = layer.input->shape[1];
                const int output_size = layer.output->shape[1];
                
                // Compute gradient w.r.t weights
                std::vector<double> weight_grad(layer.weights->data.size(), 0.0);
                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < input_size; ++i) {
                        for (int j = 0; j < output_size; ++j) {
                            weight_grad[i * output_size + j] += layer.input->data[b * input_size + i] * grad[b * output_size + j];
                        }
                    }
                }
                
                // Update weights
                for (size_t j = 0; j < layer.weights->data.size(); ++j) {
                    layer.weights->grad[j] += weight_grad[j]; // 0.01 is the learning rate
                }

                // Compute gradient w.r.t bias and update
                std::vector<double> bias_grad(output_size, 0.0);
                for (int b = 0; b < batch_size; ++b) {
                    for (int j = 0; j < output_size; ++j) {
                        bias_grad[j] += grad[b * output_size + j];
                    }
                }
                for (size_t j = 0; j < layer.bias->data.size(); ++j) {
                    layer.bias->grad[j] += bias_grad[j];
                }

                // Compute gradient w.r.t input for next layer
                std::vector<double> input_grad(batch_size * input_size, 0.0);
                for (int b = 0; b < batch_size; ++b) {
                    for (int i = 0; i < input_size; ++i) {
                        for (int j = 0; j < output_size; ++j) {
                            input_grad[b * input_size + i] += grad[b * output_size + j] * layer.weights->data[i * output_size + j];
                        }
                    }
                }
                grad = std::move(input_grad);
                break;
            }

            case LayerType::RELU: {
                // ReLU backward pass
                for (size_t j = 0; j < grad.size(); ++j) {
                    grad[j] = (layer.input->data[j] > 0) ? grad[j] : 0;
                }
                break;
            }
        }

        // Store the gradient in the input tensor of the current layer
        layer.input->grad = grad;
    }
}