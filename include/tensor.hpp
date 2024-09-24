#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <random>

class Tensor {
public:
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<int> shape;
    int ndim;
    size_t total_size;
    bool require_grad;

    // Constructors
    Tensor(const std::vector<int>& shape, bool require_grad = false);
    Tensor(const std::vector<int>& shape, bool require_grad, bool randomize);

    // Copy constructor
    Tensor(const Tensor& other);

    // Move constructor
    Tensor(Tensor&& other) noexcept;

    // Copy assignment operator
    Tensor& operator=(const Tensor& other);

    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept;

    // Destructor
    ~Tensor() = default;

    // Utility functions
    void print() const;
    Tensor reshape(const std::vector<int>& new_shape) const;
    double& operator()(const std::vector<int>& indices);
    const double& operator()(const std::vector<int>& indices) const;

    // Static factory methods
    static std::unique_ptr<Tensor> zeros(const std::vector<int>& shape);
    static std::unique_ptr<Tensor> ones(const std::vector<int>& shape);
    static std::unique_ptr<Tensor> random(const std::vector<int>& shape, double min = 0.0, double max = 1.0);

private:
    void initialize(bool randomize);
    size_t calculate_index(const std::vector<int>& indices) const;
};

#endif // TENSOR_HPP