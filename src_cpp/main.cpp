#include "../include/tensor.hpp"
#include "../include/model.hpp"
#include "../include/utils.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>

#define BATCH_SIZE 16
#define EPOCHS 75

class Dataset {
public:
    int count;
    Tensor* inputs;
    Tensor* actual;

    Dataset(int num_datapoints, int size_per_point) : count(num_datapoints) {
        int shape[] = {num_datapoints, size_per_point};
        inputs = create_tensor(shape, 2, false);
        shape[0] = 1;
        shape[1] = num_datapoints;
        actual = create_tensor(shape, 2, false);
    }

    ~Dataset() {
        free_tensor(inputs);
        free_tensor(actual);
    }
};

void MNIST_dataset(const std::string& filename, Dataset* dataset) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    int datapoint_index = 0;

    while (std::getline(file, line) && datapoint_index < dataset->count) {
        std::istringstream iss(line);
        std::string input, actual;
        
        if (std::getline(iss, input, ';') && std::getline(iss, actual)) {
            std::istringstream input_stream(input);
            std::string token;
            int i = 0;
            while (std::getline(input_stream, token, ',') && i < dataset->inputs->shape[1]) {
                dataset->inputs->data[datapoint_index * 784 + i] = std::stod(token);
                i++;
            }

            dataset->actual->data[datapoint_index] = std::stod(actual);
            datapoint_index++;
        }
    }

    file.close();
}

int main() {
    std::srand(std::time(nullptr));
    
    // Load dataset
    Dataset dataset(86184, 784);
    MNIST_dataset("data/train_dataset.txt", &dataset);

    // Create model
    Model* model = create_model(6);
    add_layer(model, LINEAR_LAYER, 784, 500);
    add_layer(model, RELU_LAYER, 500, 500);
    add_layer(model, LINEAR_LAYER, 500, 100);
    add_layer(model, RELU_LAYER, 100, 100);
    add_layer(model, LINEAR_LAYER, 100, 10);
    add_layer(model, SOFTMAX_LAYER, 10, 10);

    int num_batches = dataset.count / BATCH_SIZE;
    double learning_rate = 0.01;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;

        for (int batch = 0; batch < num_batches; batch++) {
            int input_shape[] = {BATCH_SIZE, 784};
            Tensor* input = create_tensor(input_shape, 2, false);
            
            int actual_shape[] = {BATCH_SIZE, 1};
            Tensor* y_act = create_tensor(actual_shape, 2, false);

            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = batch * BATCH_SIZE + i;
                std::memcpy(&input->data[i * 784], &dataset.inputs->data[idx * 784], 784 * sizeof(double));
                y_act->data[i] = dataset.actual->data[idx];
            }

            Tensor* pred = forward(model, input);
            double loss = cross_entropy_loss(pred, y_act);
            total_loss += loss;

            backwards(model, pred, y_act);
            SGD_step(model, learning_rate);
            zero_grad(model);

            free_tensor(input);
            free_tensor(y_act);
            free_tensor(pred);
        }

        std::cout << "Epoch " << epoch + 1 << ", Average Loss: " << total_loss / num_batches << std::endl;
    }

    Dataset test_dataset(21546, 784);
    MNIST_dataset("data/test_dataset.txt", &test_dataset);
    int correct_predictions = 0;
    int total_predictions = test_dataset.count;

    for (int i = 0; i < total_predictions; i++) {
        int input_shape[] = {1, 784};
        Tensor* input = create_tensor(input_shape, 2, false);
        
        std::memcpy(input->data, &test_dataset.inputs->data[i * 784], 784 * sizeof(double));

        Tensor* pred = forward(model, input);

        int predicted_class = 0;
        double max_prob = pred->data[0];
        for (int j = 1; j < 10; j++) {
            if (pred->data[j] > max_prob) {
                max_prob = pred->data[j];
                predicted_class = j;
            }
        }

        int actual_class = static_cast<int>(test_dataset.actual->data[i]);
        if (predicted_class == actual_class) {
            correct_predictions++;
        }

        free_tensor(input);
        free_tensor(pred);
    }

    double accuracy = static_cast<double>(correct_predictions) / total_predictions * 100.0;
    std::cout << "Model Accuracy: " << accuracy << "%" << std::endl;

    free_model(model);

    return 0;
}