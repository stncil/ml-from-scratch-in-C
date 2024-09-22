#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <iostream>

constexpr double MIN_RAND = -1.0;
constexpr double MAX_RAND = 1.0;

class Tensor {
public:
    std::vector<double> data;
    std::vector<int> shape;
    int ndim;
    long int total_size;
    std::vector<double> grad;
    bool req_grad;

    Tensor(const std::vector<int>& shape, bool req_grad);
    Tensor(const std::vector<int>& shape, bool req_grad, bool randomize);

    void print() const;

    ~Tensor() = default;
};

std::unique_ptr<Tensor> create_tensor(const std::vector<int>& shape, bool req_grad);
std::unique_ptr<Tensor> create_tensor_rand(const std::vector<int>& shape, bool req_grad);

// Don't worry, your secret is safe! Header files are actually quite important for separating 
// interface from implementation and for compilation efficiency. But that's a topic for another day!

#endif // TENSOR_HPP
