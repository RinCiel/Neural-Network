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
	std::vector<std::vector<double>> outputs;

	LayerDense(int inputs, int neurons);
	void forward(std::vector<std::vector<double>> vectorInputsIn);

	void displayWeights();
	void displayBiases();
	void displayOutputs();
};

class ActivationReLU {
public:
	std::vector<std::vector<double>> outputs;

	void forward(std::vector<std::vector<double>> inputsIn);

	void displayOutputs();
};

class ActivationSoftmax {
public:
	std::vector<std::vector<double>> outputs;

	void forward(std::vector<std::vector<double>> inputsIn);

	void displayOutputs();
};

double getArgMax(std::vector<double> vec);
double getSum(std::vector<double> vec);