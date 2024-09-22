#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "tensor.hpp"
#include <memory>

class Operations {
public:
    static std::unique_ptr<Tensor> matmul(const Tensor& t1, const Tensor& t2);
    static void matmul_d(Tensor& t1, const Tensor& t2, const Tensor& prev_layer_grad);
    static std::unique_ptr<Tensor> add_bias(const Tensor& t1, const Tensor& bias);
    static std::unique_ptr<Tensor> relu(const Tensor& t1);
    static void relu_d(Tensor& input, const Tensor& grad_output);
    static std::unique_ptr<Tensor> softmax(const Tensor& t1);
};

#endif // OPERATIONS_HPP
