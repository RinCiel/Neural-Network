/*
Neural Network Library 
Based on Neural Networks from Scratch in Python by Harrison Kinsley

Call NeuralNetworkInit() first
*/
#include "NeuralNetwork.h"

void NeuralNetworkInit() {
	// set random seed
	srand(time(NULL));
}

LayerDense::LayerDense(int inputs, int neurons) {
	// fill weights with random numbers between -1 and 1 
	// weights size: (inputs, neurons)
	for (int i = 0; i < inputs; i++) {
		std::vector<double> row;
		for (int j = 0; j < neurons; j++) {
			row.push_back((double)rand() / RAND_MAX * 2 - 1);
		}
		weights.push_back(row);
	}

	// fill biases with 0
	// biases size: (neurons)
	for (int i = 0; i < neurons; i++) {
		biases.push_back(0);
	}
}

void LayerDense::forward(std::vector<std::vector<double>> inputs) {
	// multiply inputs with weights
	output = dot(inputs, weights);

	// add biases
	for (int i = 0; i < output.size(); i++) {
		for (int j = 0; j < output[i].size(); j++) {
			output[i][j] += biases[j];
		}
	}
}

void Activation_ReLU::forward(std::vector<std::vector<double>> inputs) {
	// apply ReLU activation function
	output = inputs;
	for (int i = 0; i < output.size(); i++) {
		for (int j = 0; j < output[i].size(); j++) {
			output[i][j] = std::max(0.0, output[i][j]);
		}
	}
}

void Activation_Softmax::forward(std::vector<std::vector<double>> inputs) {
	// apply softmax activation function
	output = inputs;
	for (int i = 0; i < output.size(); i++) {
		double sum = getSum(output[i]);
		for (int j = 0; j < output[i].size(); j++) {
			output[i][j] = exp(output[i][j]) / sum;
		}
	}
}

void Loss_CategoricalCrossEntropy::forward(std::vector<std::vector<double>> inputs, std::vector<double> targets) {
	std::vector<double> allLoss;
	std::vector<double> current;
	for (int i = 0; i < inputs.size(); i++) {
		current = clip(1e-7, 1 - 1e-7, inputs[i]);
		allLoss.push_back(-1 * log(current[targets[i]]));
	}
	output = getSum(allLoss) / allLoss.size();
}

// overload the forward function to account for 2d vectors
void Loss_CategoricalCrossEntropy::forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
	std::vector<double> current;
	std::vector<double> allLoss;
	for (int i = 0; i < inputs.size(); i++) {
		current = clip(1e-7, 1 - 1e-7, inputs[i]);
		for (int j = 0; j < current.size(); j++) {
			current[j] = current[j] * targets[i][j];
		}
		allLoss.push_back(-1 * log(getSum(current)));
		current.clear();
 	}
	output = getSum(allLoss) / allLoss.size();
};

// checks the target outputs with confidence
// if the highest confidence equals target output index, then it is accurate
void Accuracy::forward(std::vector<std::vector<double>> inputs, std::vector<double> targets) {
	double res = 0;
	for (int i = 0; i < inputs.size(); i++) {
		if (getArgMax(inputs[i]) == targets[i]) {
			res++;
		}
	}
	output = res / targets.size();
}

void Accuracy::forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
	double res = 0;
	for (int i = 0; i < inputs.size(); i++) {
		if (getArgMax(inputs[i]) == getArgMax(targets[i])) {
			res++;
		}
	}
	output = res / targets.size();
}