#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <algorithm>
#include "utils.h"

void NeuralNetworkInit();

class LayerDense {
    public:
        LayerDense(int inputs, int neurons);
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;

        void forward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> output;
};

class Activation_ReLU {
    public:
        void forward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> output;
};

class Activation_Softmax {
    public:
        void forward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> output;
};

class Loss_CategoricalCrossEntropy {
    public:
        void forward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
        void forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
        double output;
};

class Accuracy {
    public:
        void forward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
        void forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
        double output;
};