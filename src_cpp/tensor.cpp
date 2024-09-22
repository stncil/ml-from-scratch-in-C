#include "../include/tensor.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <random>

// prints
void print_tensor(Tensor* tensor) {
    std::cout << "Tensor shape: (";
    for (int i = 0; i < tensor->ndim; i++) {
        std::cout << tensor->shape[i];
        if (i < tensor->ndim - 1) std::cout << ", ";
    }
    std::cout << ")\n";

    std::cout << "Data: ";
    if (tensor->ndim == 1) {
        std::cout << "[ ";
        for (int i = 0; i < tensor->total_size; i++) {
            std::cout << std::fixed << std::setprecision(4) << tensor->data[i] << " ";
        }
        std::cout << "]";
    } else if (tensor->ndim == 2) {
        std::cout << "[ \n";
        for (int i = 0; i < tensor->shape[0]; i++) {
            std::cout << "[ ";
            for (int j = 0; j < tensor->shape[1]; j++) {
                std::cout << std::fixed << std::setprecision(10) << tensor->data[i * tensor->shape[1] + j] << " ";
            }
            std::cout << "]\n";
        }
        std::cout << "]";
    } else {
        std::cout << "[ ";
        for (int i = 0; i < tensor->total_size; i++) {
            std::cout << std::fixed << std::setprecision(4) << tensor->data[i] << " ";
        }
        std::cout << "]";
    }

    std::cout << std::endl;
}

// makes a tensor initialized to 0
Tensor* create_tensor(const int* shape, int ndim, bool req_grad) {
    if (ndim < 0) {
        std::cerr << "Failed to create tensor, dim is less than 0: " << ndim << std::endl;
        return nullptr;
    }

    Tensor* tensor = new Tensor;
    if (tensor == nullptr) {
        std::cerr << "Failed to allocate space for tensor" << std::endl;
        exit(EXIT_FAILURE);
    }
    tensor->shape = new int[ndim];
    if (tensor->shape == nullptr) {
        std::cerr << "Failed to allocate space for shape of tensor" << std::endl;
        exit(EXIT_FAILURE);
    }
    tensor->ndim = ndim;
    
    unsigned int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        total_size *= shape[i];
    }
    tensor->total_size = total_size;

    tensor->data = new double[total_size]();
    if (tensor->data == nullptr) {
        std::cerr << "Failed to allocate space for data of tensor" << std::endl;
        exit(EXIT_FAILURE);
    }

    tensor->req_grad = req_grad;
    if (req_grad) {
        tensor->grad = new double[total_size]();
        if (tensor->grad == nullptr) {
            std::cerr << "Failed to allocate space for grad of tensor" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return tensor;
}

// Tensor values randomized
Tensor* create_tensor_rand(const int* shape, int ndim, bool req_grad) {
    if (ndim < 0) {
        std::cerr << "Failed to create tensor, dim is less than 0: " << ndim << std::endl;
        return nullptr;
    }

    Tensor* tensor = new Tensor;
    if (tensor == nullptr) {
        std::cerr << "Failed to allocate space for tensor" << std::endl;
        exit(EXIT_FAILURE);
    }
    tensor->shape = new int[ndim];
    if (tensor->shape == nullptr) {
        std::cerr << "Failed to allocate space for shape of tensor" << std::endl;
        exit(EXIT_FAILURE);
    }
    tensor->ndim = ndim;
    
    unsigned int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        total_size *= shape[i];
    }
    tensor->total_size = total_size;

    tensor->data = new double[total_size];
    if (tensor->data == nullptr) {
        std::cerr << "Failed to allocate space for data of tensor" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(MIN_RAND, MAX_RAND);

    for (unsigned int i = 0; i < total_size; i++) {
        tensor->data[i] = dis(gen);
    }

    tensor->req_grad = req_grad;
    if (req_grad) {
        tensor->grad = new double[total_size]();
        if (tensor->grad == nullptr) {
            std::cerr << "Failed to allocate space for grad of tensor" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return tensor;
}

void free_tensor(Tensor* tensor) {
    delete[] tensor->data;
    delete[] tensor->shape;
    if (tensor->req_grad) {
        delete[] tensor->grad;
    }
    delete tensor;
}