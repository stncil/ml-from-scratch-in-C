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
        layer->input = std::make_unique<Tensor>(x->shape, true, true);
        std::copy(x->data.begin(), x->data.end(), layer->input->data.begin());

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
                                                             x->data.begin() + ((b + 1) * (class_count)));
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

void backward(Model& model, Tensor& pred, const Tensor& actual) {
    int last_layer = model.layers.size() - 1;
    const int batch_size = pred.shape[0];

    // Compute initial gradient (assuming cross-entropy loss with softmax output)
    std::vector<double> grad = pred.grad;
    for (size_t i = 0; i < actual.data.size(); ++i) {
        // pred.grad[i] = pred.data[i] - (i % pred.shape[1] == static_cast<int>(actual.data[i / pred.shape[1]]));
        grad[i] = pred.grad[i];
    }

    for (int i = last_layer; i >= 0; --i) {
        // Layer& layer = *model.layers[i];

        switch (model.layers[i]->layer_type) {
            case LayerType::SOFTMAX:
                // Softmax gradient is already computed in the initial step
                break;

            case LayerType::LINEAR: {
                const int input_size = model.layers[i]->input->shape[1];
                const int output_size = model.layers[i]->output->shape[1];
                
                // Compute gradient w.r.t weights
                std::vector<double> weight_grad(model.layers[i]->weights->data.size(), 0.0);
                for (int b = 0; b < batch_size; ++b) {
                    for (int index = 0; index < input_size; ++index) {
                        for (int j = 0; j < output_size; ++j) {
                            weight_grad[index * output_size + j] += model.layers[i]->input->data[b * input_size + index] * grad[b * output_size + j];
                        }
                    }
                }
                
                // Update weights
                for (size_t j = 0; j < model.layers[i]->weights->data.size(); ++j) {
                    model.layers[i]->weights->grad[j] += weight_grad[j]; // 0.01 is the learning rate
                }

                // Compute gradient w.r.t bias and update
                std::vector<double> bias_grad(output_size, 0.0);
                for (int b = 0; b < batch_size; ++b) {
                    for (int j = 0; j < output_size; ++j) {
                        bias_grad[j] += grad[b * output_size + j];
                    }
                }
                for (size_t j = 0; j < model.layers[i]->bias->data.size(); ++j) {
                    model.layers[i]->bias->grad[j] += bias_grad[j];
                }

                // Compute gradient w.r.t input for next layer
                std::vector<double> input_grad(batch_size * input_size, 0.0);
                for (int b = 0; b < batch_size; ++b) {
                    for (int index = 0; index < input_size; ++index) {
                        for (int j = 0; j < output_size; ++j) {
                            input_grad[b * input_size + index] += grad[b * output_size + j] * model.layers[i]->weights->data[index * output_size + j];
                        }
                    }
                }
                grad = std::move(input_grad);
                break;
            }

            case LayerType::RELU: {
                // ReLU backward pass
                // for (size_t j = 0; j < grad.size(); ++j) {
                    std::transform(model.layers[i]->input->data.begin(), model.layers[i]->input->data.end(), grad.begin(), grad.begin(),
               [](double input, double grad) { return input > 0 ? grad : 0; });
               
                // }
                break;
            }
        }

        // Store the gradient in the input tensor of the current layer
        model.layers[i]->input->grad = grad;
    }
}