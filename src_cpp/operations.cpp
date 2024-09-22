#include "../include/tensor.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// The matmul operation which is VERY slow and I will def speed up either from scratch or just yoink BLAS, also only up to 2 dim for right now
Tensor* matmul(Tensor* t1, Tensor* t2) {
    if (t1->ndim > 2 || t2->ndim > 2) {
        throw std::runtime_error("Mat mul currently only supports 2 dimensional matrices");
    }

    if (t1->shape[1] != t2->shape[0]) {
        throw std::runtime_error("Invalid dimensions for mat mul: " + 
                                 std::to_string(t1->shape[0]) + "x" + std::to_string(t1->shape[1]) + " * " +
                                 std::to_string(t2->shape[0]) + "x" + std::to_string(t2->shape[1]));
    }
    
    int m = t1->shape[0];
    int n = t1->shape[1];
    int p = t2->shape[1];

    int shape[] = {m, p};

    Tensor* res = create_tensor(shape, 2, t1->req_grad || t2->req_grad);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += t1->data[i * n + k] * t2->data[k * p + j];
            }
            res->data[i * p + j] = sum;
        }
    }
    return res;
}

void matmul_d(Tensor* t1, Tensor* t2, Tensor* prev_layer_grad) {
    int m = t1->shape[0];
    int n = t1->shape[1];
    int p = t2->shape[1];
    
    // t1.grad = Prev times t2^T
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < p; k++) {
                sum += prev_layer_grad->grad[i * p + k] * t2->data[j * p + k];
            }
            t1->grad[i * n + j] += sum;
        }
    }

    // t2.grad = t1^T times prev
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < p; k++) {
            double sum = 0;
            for (int i = 0; i < m; i++) {
                sum += t1->data[i * n + j] * prev_layer_grad->grad[i * p + k];
            }
            t2->grad[j * p + k] += sum;
        }
    }
}

Tensor* add_bias(Tensor* t1, Tensor* bias) {
    if (t1->ndim > 2) {
        throw std::runtime_error("Adding bias currently only supports 2 dim tensors");
    }
    if (t1->ndim < 1 || bias->ndim != 1 || t1->shape[1] != bias->shape[0]) {
        throw std::runtime_error("Invalid shapes for adding biases: adding dim - 1: " + 
                                 std::to_string(t1->shape[1]) + " with bias " + std::to_string(bias->shape[0]));
    }

    Tensor* output = create_tensor(t1->shape, t1->ndim, t1->req_grad);
    for (int batch = 0; batch < t1->shape[0]; batch++) {
        for (int i = 0; i < t1->shape[1]; i++) {
            output->data[batch * t1->shape[1] + i] = bias->data[i] + t1->data[batch * t1->shape[1] + i];
        }
    }
    return output;
}

Tensor* relu(Tensor* t1) {
    Tensor* res = create_tensor(t1->shape, t1->ndim, t1->req_grad);
    if (t1->total_size != res->total_size) {
        throw std::runtime_error("Incompatible sizes for relu input: " + 
                                 std::to_string(t1->total_size) + " and output: " + std::to_string(res->total_size));
    }
    for (int i = 0; i < t1->total_size; i++) {
        res->data[i] = std::max(0.0, t1->data[i]);
    }
    return res;
}

void relu_d(Tensor* input, Tensor* grad_output) {
    for (int i = 0; i < input->total_size; i++) {
        input->grad[i] = grad_output->grad[i] * (input->data[i] > 0);
    }
}

Tensor* softmax(Tensor* t1) {
    Tensor* res = create_tensor(t1->shape, t1->ndim, t1->req_grad);

    int batch_size = t1->shape[0];
    int size = t1->shape[1];
    for (int b = 0; b < batch_size; b++) {
        double max = *std::max_element(t1->data + size * b, t1->data + size * (b + 1));
        
        double sum = 0;
        for (int i = 0; i < size; i++) {
            res->data[size * b + i] = std::exp(t1->data[size * b + i] - max);
            sum += res->data[size * b + i];
        }

        for (int i = 0; i < size; i++) {
            res->data[size * b + i] /= sum;
        }
    }

    return res;
}