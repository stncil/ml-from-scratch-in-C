#include "../include/model.hpp"
#include "../include/operations.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>

void add_layer(Model* model, LAYER_TYPE type, int input, int output) {
    if (model->layer_index >= model->layer_size) {
        throw std::runtime_error("Trying to put a layer at index more than layer size: index " + 
                                 std::to_string(model->layer_index) + ", size " + std::to_string(model->layer_size));
    }

    model->layers[model->layer_index].layer_type = type;

    if (type == LINEAR_LAYER) {
        int shape[] = {input, output};
        model->layers[model->layer_index].weights = create_tensor_rand(shape, 2, true);
        int bias_shape[] = {output};
        model->layers[model->layer_index].bias = create_tensor(bias_shape, 1, true);
    } else {
        model->layers[model->layer_index].weights = nullptr;
        model->layers[model->layer_index].bias = nullptr;
    }

    model->layers[model->layer_index].output = nullptr;
    model->layers[model->layer_index].input = nullptr;

    model->layer_index++;
}

Model* create_model(int num_layers) {
    Model* model = new Model;
    if (model == nullptr) {
        throw std::runtime_error("Unable to allocate enough memory when creating a model");
    }
    model->layer_index = 0;
    model->layers = new Layer[num_layers]();
    if (model->layers == nullptr) {
        throw std::runtime_error("Unable to allocate enough memory when creating initializing " + std::to_string(num_layers) + " layers");
    }
    model->layer_size = num_layers;

    return model;
}

Tensor* forward(Model* model, Tensor* input) {
    Tensor* x = input;

    for (int i = 0; i < model->layer_size; i++) {
        Layer* cur_layer = &model->layers[i];
        Tensor* next;

        if (cur_layer->input) {
            free_tensor(cur_layer->input);
        }
        cur_layer->input = create_tensor(x->shape, x->ndim, true);
        std::memcpy(cur_layer->input->data, x->data, x->shape[0] * x->shape[1] * sizeof(double));

        switch (cur_layer->layer_type) {
            case LINEAR_LAYER:
                next = matmul(x, cur_layer->weights);
                for (int j = 0; j < next->shape[1]; j++) {
                    next->data[j] += cur_layer->bias->data[j];
                }
                break;
            case RELU_LAYER:
                next = relu(x);
                break;
            case SOFTMAX_LAYER:
                next = softmax(x);
                break;
            default:
                throw std::runtime_error("INVALID INPUT FOR LAYER DURING FPASS " + std::to_string(cur_layer->layer_type));
        }

        cur_layer->output = next;
        if (i > 0) {
            free_tensor(x);
        }
        x = next;
    }

    return x;
}

void backwards(Model* model, Tensor* pred, Tensor* act) {
    int last_layer = model->layer_size - 1;

    cross_entropy_softmax_backwards(pred, pred, act);

    Tensor* cur_grad = pred;
    for (int i = last_layer; i >= 0; i--) {
        Layer* cur_layer = &model->layers[i];

        switch (cur_layer->layer_type) {
            case SOFTMAX_LAYER:
                break;
            case LINEAR_LAYER:
                matmul_d(cur_layer->input, cur_layer->weights, cur_grad);

                for (int j = 0; j < cur_layer->bias->total_size; j++) {
                    double sum = 0.0;
                    for (int batch = 0; batch < cur_grad->shape[0]; batch++) {
                        sum += cur_grad->grad[batch * cur_layer->bias->total_size + j];
                    }
                    cur_layer->bias->grad[j] += sum;
                }

                cur_grad = cur_layer->input;
                break;
            case RELU_LAYER:
                relu_d(cur_layer->input, cur_grad);
                cur_grad = cur_layer->input;
                break;
            default:
                throw std::runtime_error("INVALID INPUT FOR BACKWARDS: " + std::to_string(cur_layer->layer_type));
        }
    }
}

void free_layer(Layer *layer) {
    if (layer->bias) {
        free_tensor(layer->bias);
    }
    if (layer->input) {
        free_tensor(layer->input);
    }
    if (layer->weights) {
        free_tensor(layer->weights);
    }
}

void free_model(Model* model) {
    for (int layer = 0; layer < model->layer_size; layer++) {
        free_layer(&model->layers[layer]);
    }
    delete[] model->layers;
    delete model;
}
