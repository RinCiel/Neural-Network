#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>

void NeuralNetworkInit();

class LayerDense {
public:
	std::vector<std::vector<double>> vectorInputs;
	int inputs;
	int neurons;
	std::vector<std::vector<double>> layerWeights;
	std::vector<double> layerBiases;
	std::vector<std::vector<double>> output;

	LayerDense(int inputs, int neurons);
	void forward(std::vector<std::vector<double>> vectorInputsIn);

	void displayWeights();
	void displayBiases();
	void displayOutputs();
};

class ActivationReLU {
public:
	std::vector<std::vector<double>> output;

	void forward(std::vector<std::vector<double>> inputsIn);

	void displayOutputs();
};

class ActivationSoftmax {
public:
	std::vector<std::vector<double>> output;

	void forward(std::vector<std::vector<double>> inputsIn);

	void displayOutputs();
};

class LossCategoricalCrossEntropy {
public:
	std::vector<double> targetOutputs;
	double output;

	void forward(std::vector<std::vector<double>> inputs, std::vector<double> targets);
	void forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets);
};

double getArgMax(std::vector<double> vec);
double getSum(std::vector<double> vec);
std::vector<double> clip(double min, double max, std::vector<double> vec);