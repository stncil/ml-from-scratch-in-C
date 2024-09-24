#include "../include/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <random>

Tensor::Tensor(const std::vector<int>& shape, bool require_grad)
    : shape(shape), ndim(shape.size()), require_grad(require_grad) {
    total_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    data.resize(total_size, 0.0);
    if (require_grad) {
        grad.resize(total_size, 0.0);
    }
}

Tensor::Tensor(const std::vector<int>& shape, bool require_grad, bool randomize)
    : Tensor(shape, require_grad) {
    initialize(randomize);
}

Tensor::Tensor(const Tensor& other)
    : data(other.data), grad(other.grad), shape(other.shape),
      ndim(other.ndim), total_size(other.total_size), require_grad(other.require_grad) {}

Tensor::Tensor(Tensor&& other) noexcept
    : data(std::move(other.data)), grad(std::move(other.grad)), shape(std::move(other.shape)),
      ndim(other.ndim), total_size(other.total_size), require_grad(other.require_grad) {}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        data = other.data;
        grad = other.grad;
        shape = other.shape;
        ndim = other.ndim;
        total_size = other.total_size;
        require_grad = other.require_grad;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        data = std::move(other.data);
        grad = std::move(other.grad);
        shape = std::move(other.shape);
        ndim = other.ndim;
        total_size = other.total_size;
        require_grad = other.require_grad;
    }
    return *this;
}

void Tensor::initialize(bool randomize) {
    if (randomize) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1.0, 1.0);
        std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
    } else {
        std::fill(data.begin(), data.end(), 0.0);
    }
}

void Tensor::print() const {
    std::cout << "Tensor shape: (";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")\n";

    std::cout << "Data:\n";
    // This is a simplified print for all dimensions
    // You might want to implement a more sophisticated printing for multi-dimensional tensors
    for (const auto& val : data) {
        std::cout << std::setprecision(4) << std::fixed << val << " ";
    }
    std::cout << std::endl;
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    size_t new_total_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_total_size != total_size) {
        throw std::invalid_argument("New shape is incompatible with the current data size");
    }
    Tensor reshaped(new_shape, require_grad);
    reshaped.data = data;
    if (require_grad) {
        reshaped.grad = grad;
    }
    return reshaped;
}

double& Tensor::operator()(const std::vector<int>& indices) {
    return data[calculate_index(indices)];
}

const double& Tensor::operator()(const std::vector<int>& indices) const {
    return data[calculate_index(indices)];
}

size_t Tensor::calculate_index(const std::vector<int>& indices) const {
    if (indices.size() != ndim) {
        throw std::invalid_argument("Number of indices does not match tensor dimensions");
    }
    size_t index = 0;
    size_t multiplier = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= shape[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        index += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return index;
}

std::unique_ptr<Tensor> Tensor::zeros(const std::vector<int>& shape) {
    return std::make_unique<Tensor>(shape, false);
}

std::unique_ptr<Tensor> Tensor::ones(const std::vector<int>& shape) {
    auto tensor = std::make_unique<Tensor>(shape, false);
    std::fill(tensor->data.begin(), tensor->data.end(), 1.0);
    return tensor;
}

std::unique_ptr<Tensor> Tensor::random(const std::vector<int>& shape, double min, double max) {
    auto tensor = std::make_unique<Tensor>(shape, false, true);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    std::generate(tensor->data.begin(), tensor->data.end(), [&]() { return dis(gen); });
    return tensor;
}