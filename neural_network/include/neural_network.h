#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include <algorithm>
#include <cmath>

#include "vector_utils.h"

void NeuralNetworkInit();

class Layer_Dense {
    public:
        Layer_Dense(int inputs, int neurons);
        std::vector<std::vector<double>> weights;
        std::vector<double> biases;

        void forward(std::vector<std::vector<double>> inputs, bool normalize=false);
        std::vector<std::vector<double>> input;
        std::vector<std::vector<double>> output;

        void backward(std::vector<std::vector<double>> inputs);
        std::vector<std::vector<double>> dWeights;
        std::vector<double> dBiases;
        std::vector<std::vector<double>> dInputs;

        bool momentum_set = false;
        std::vector<std::vector<double>> momentum_weights;
        std::vector<double> momentum_biases;

        bool cache_set = false;
        std::vector<std::vector<double>> cache_weights;
        std::vector<double> cache_biases;
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
        Optimizer_SGD(double learning_rate=1.0, double decay=0.0, double momentum=0.0);

        int iterations; 

        double learning_rate;
        double current_learning_rate;

        void update(Layer_Dense* layer);

        void applyDecay_pre();
        void applyDecay_post();
        double decay;

        void applyMomentum(Layer_Dense* layer);
        double momentum;
        std::vector<std::vector<double>> momentum_weights;
        std::vector<double> momentum_biases;
};

class Optimizer_Adagrad {
    public:
        Optimizer_Adagrad(double learning_rate=1.0, double decay=0.0, double epsilon=1e-7);
        double learning_rate;
        double current_learning_rate;
        double decay;
        double epsilon;
        int iterations;

        void update(Layer_Dense* layer);

        void applyDecay_pre();
        void applyDecay_post();
};
