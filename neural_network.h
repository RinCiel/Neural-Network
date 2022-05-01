#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <algorithm>
#include "vector_utils.h"

void NeuralNetworkInit();

class LayerDense {
    public:
        LayerDense(int inputs, int neurons);
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;

        void forward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> input;
        std::vector<std::vector<double>> output;

        void backward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> dWeights;
        std::vector<double> dBiases;
        std::vector<std::vector<double>> dInputs;
};

class Activation_ReLU {
    public:
        void forward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> output;

        void backward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> dInputs;
};

class Activation_Softmax {
    public:
        void forward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> output;

        void backward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> dInputs;
};

class Loss_CategoricalCrossEntropy {
    public:
        void forward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
        void forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
        double output;

        void backward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
        void backward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
        std::vector<std::vector<double>> dInputs;
};

class Accuracy {
    public:
        void forward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
        void forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
        double output;
};

class Activation_Softmax_Loss_CategoricalCrossEntropy {
    public:
        Activation_Softmax_Loss_CategoricalCrossEntropy();
        void forward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
        Activation_Softmax* softmax;
        Loss_CategoricalCrossEntropy* loss;
        std::vector<std::vector<double>> output;

        void backward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
        void backward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
        std::vector<std::vector<double>> dInputs;
};

class Optimizer_SGD {
    public:
        Optimizer_SGD(double learning_rate=1.0);
        double learning_rate;

        void update(LayerDense* layer);
};