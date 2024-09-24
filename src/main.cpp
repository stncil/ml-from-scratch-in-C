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
#include <algorithm>


class Dataset {
public:
    int count;
    std::unique_ptr<Tensor> inputs;
    std::unique_ptr<Tensor> actual;

    Dataset(int num_datapoints, int size_per_point) : count(num_datapoints) {
        std::vector<int> shape = {num_datapoints, size_per_point};
        inputs = std::make_unique<Tensor>(shape, 2, false);
        shape[0] = 1;
        shape[1] = num_datapoints;
        actual = std::make_unique<Tensor>(shape, 2, false);
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
    Model model(6);
    model.add_layer(LayerType::LINEAR, 784, 500);
    model.add_layer(LayerType::RELU, 500, 500);
    model.add_layer(LayerType::LINEAR, 500, 100);
    model.add_layer(LayerType::RELU, 100, 100);
    model.add_layer(LayerType::LINEAR, 100, 10);
    model.add_layer(LayerType::SOFTMAX, 10, 10);


    Utils utility;

    const int BATCH_SIZE = 32;  // Assuming this is defined
    const int EPOCHS = 10;      // Assuming this is defined
    int num_batches = dataset.count / BATCH_SIZE;
    double learning_rate = 0.01;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_loss = 0.0;

        for (int batch = 0; batch < num_batches; batch++) {
            auto input = std::make_unique<Tensor>(std::vector<int>{BATCH_SIZE, 784}, false);
            auto y_act = std::make_unique<Tensor>(std::vector<int>{BATCH_SIZE, 1}, false);

            for (int i = 0; i < BATCH_SIZE; i++) {
                int idx = batch * BATCH_SIZE + i;
                std::memcpy(&input->data[i * 784], &dataset.inputs->data[idx * 784], 784 * sizeof(double));
                y_act->data[i] = dataset.actual->data[idx];
            }

            auto pred = forward(model, *input);
            // std::cout << pred->data.size();
            double loss = utility.cross_entropy_loss(*pred, *y_act);
            total_loss += loss;

            backward(model, *pred, *y_act);
            utility.SGD_step(&model, learning_rate);
            utility.zero_grad(&model);
        }

        std::cout << "Epoch " << epoch + 1 << ", Average Loss: " << total_loss / num_batches << std::endl;
    }

    // Evaluation
    Dataset test_dataset(21546, 784);
    MNIST_dataset("data/test_dataset.txt", &test_dataset);
    int correct_predictions = 0;
    int total_predictions = test_dataset.count;

    for (int i = 0; i < total_predictions; i++) {
        auto input = std::make_unique<Tensor>(std::vector<int>{1, 784}, false);
        std::memcpy(input->data.data(), &test_dataset.inputs->data[i * 784], 784 * sizeof(double));

        auto pred = forward(model, *input);

        int predicted_class = std::distance(pred->data.begin(), 
                                            std::max_element(pred->data.begin(), pred->data.end()));
        int actual_class = static_cast<int>(test_dataset.actual->data[i]);
        
        if (predicted_class == actual_class) {
            correct_predictions++;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / total_predictions * 100.0;
    std::cout << "Model Accuracy: " << accuracy << "%" << std::endl;

    return 0;

    return 0;
}