#ifndef UTILS_HPP
#define UTILS_HPP

#include "tensor.hpp"
#include "model.hpp"

class Utils {
public:
    static double cross_entropy_loss(const Tensor& y_pred, const Tensor& y_act);
    static void cross_entropy_softmax_backwards(Tensor& input, Tensor& output, const Tensor& actual);
    static void SGD_step(Model* model, double learning_rate);
    static void zero_grad(Model* model);
};

#endif // UTILS_HPP
