#include "../include/utils.hpp"
#include "../include/model.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>



double Utils::cross_entropy_loss(const Tensor& y_pred, const Tensor& y_act) {
    int batch_size = y_pred.shape[0];

    if (y_act.shape[1] != 1) {
        throw std::runtime_error("Actual must be of shape (n, 1)");
    }
    if (batch_size != y_act.total_size) {
        throw std::runtime_error("Invalid dims for cross entropy loss. Predicted: " + 
                                 std::to_string(batch_size) + " Actual: " + std::to_string(y_act.total_size));
    }
    int size = y_pred.shape[1];
    double loss = 0.0;

    for (int b = 0; b < batch_size; b++) {
        int true_class = static_cast<int>(y_act.data[b]);
        double pred = y_pred.data[b * size + true_class];
        loss -= std::log(std::max(pred, 1e-7));
    }

    return loss / batch_size;
}

void Utils::cross_entropy_softmax_backwards(Tensor& input, Tensor& output, const Tensor& actual) {
    int batch_size = input.shape[0];
    int size = input.shape[1];

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < size; i++) {
            double targ = (i == static_cast<int>(actual.data[b])) ? 1.0 : 0.0;
            input.grad[b * size + i] = (output.data[b * size + i] - targ) / batch_size;
        }
    }
}

void Utils::SGD_step(Model* model, double learning_rate) {
    for (auto& layer : model->layers) {
        if (layer->layer_type == LayerType::LINEAR) {
            for (size_t j = 0; j < layer->weights->total_size; j++) {
                layer->weights->data[j] -= layer->weights->grad[j] * learning_rate;
            }

            for (size_t j = 0; j < layer->bias->total_size; j++) {
                layer->bias->data[j] -= layer->bias->grad[j] * learning_rate;
            }
        }
    }
}

void Utils::zero_grad(Model* model) {
    for (auto& layer : model->layers) {
        if (layer->weights) {
            std::fill(layer->weights->grad.begin(), layer->weights->grad.end(), 0.0);
        }
        if (layer->bias) {
            std::fill(layer->bias->grad.begin(), layer->bias->grad.end(), 0.0);
        }
    }
}