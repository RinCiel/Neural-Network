/*
Neural Network Library 
Based on Neural Networks from Scratch in Python by Harrison Kinsley

Call NeuralNetworkInit() first
*/
#include "..\include\neural_network.h"

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
	input = inputs;
	// multiply inputs with weights
	output = dot(inputs, weights);

	// add biases
	for (int i = 0; i < output.size(); i++) {
		for (int j = 0; j < output[i].size(); j++) {
			output[i][j] += biases[j];
		}
	}
}

// inputs - derivative
void LayerDense::backward(std::vector<std::vector<double>> inputs) {
	dWeights = dot(transpose(input), inputs);
	dBiases = sumVertical(inputs);
	dInputs = dot(inputs, transpose(weights));
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

void Activation_ReLU::backward(std::vector<std::vector<double>> inputs) {
	// derivative of ReLU activation function
	dInputs = inputs;
	for (int i = 0; i < dInputs.size(); i++) {
		for (int j = 0; j < dInputs[i].size(); j++) {
			if (output[i][j] <= 0) {
				dInputs[i][j] = 0;
			}
		}
	}
}

void Activation_Softmax::forward(std::vector<std::vector<double>> inputs) {
	// apply softmax activation function
	output = inputs;
	for (int i = 0; i < output.size(); i++) {
		double max = getMax(output[i]);
		for (int j = 0; j < output[i].size(); j++) {
			output[i][j] = exp(output[i][j] - max);
		}

		// normalize
		double sum = getSum(output[i]);
		for (int j = 0; j < output[i].size(); j++) {
			output[i][j] /= sum;
		}
	}
}

void Activation_Softmax::backward(std::vector<std::vector<double>> inputs) {
	// fill dInput with 0 the size of inputs
	dInputs = inputs;
	for (int i = 0; i < dInputs.size(); i++) {
		for (int j = 0; j < dInputs[i].size(); j++) {
			dInputs[i][j] = 0;
		}
	}

	for (int i = 0; i < output.size(); i++) {
		std::vector<std::vector<double>> single_output = transpose(output[i]);
		// calculate jacobian matrix
		std::vector<std::vector<double>> jacobian = subtract(diagflat(single_output), dot(single_output, transpose(single_output)));
		dInputs[i] = dot(jacobian, inputs[i]);
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

void Loss_CategoricalCrossEntropy::backward(std::vector<std::vector<double>> inputs, std::vector<double> targets) {
	int samples = inputs.size();
	int labels = inputs[0].size();

	// change targets to one-hot encoding
	std::vector<std::vector<double>> targets_one_hot;
	for (int i = 0; i < samples; i++) {
		std::vector<double> row;
		for (int j = 0; j < labels; j++) {
			if (j == targets[i]) {
				row.push_back(1);
			}
			else {
				row.push_back(0);
			}
		}
		targets_one_hot.push_back(row);
	}

	this->backward(inputs, targets_one_hot);
}

void Loss_CategoricalCrossEntropy::backward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
	int samples = inputs.size();
	dInputs = divide(multiply(targets, -1), inputs);
	dInputs = divide(dInputs, samples);
}

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

Activation_Softmax_Loss_CategoricalCrossEntropy::Activation_Softmax_Loss_CategoricalCrossEntropy() {
	softmax = new Activation_Softmax();
	loss = new Loss_CategoricalCrossEntropy();
}

void Activation_Softmax_Loss_CategoricalCrossEntropy::forward(std::vector<std::vector<double>> inputs, std::vector<double> targets) {
	softmax->forward(inputs);
	loss->forward(softmax->output, targets);
	output = softmax->output;
}

void Activation_Softmax_Loss_CategoricalCrossEntropy::backward(std::vector<std::vector<double>> inputs, std::vector<double> targets) {
	int samples = inputs.size();
	dInputs = inputs;
	for (int i = 0; i < samples; i++) {
		dInputs[i][targets[i]] -= 1;
	}

	// normalize
	dInputs = divide(dInputs, samples);
}

void Activation_Softmax_Loss_CategoricalCrossEntropy::backward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
	int samples = inputs.size();
	dInputs = inputs;
	for (int i = 0; i < samples; i++) {
		dInputs[i][getArgMax(targets[i])] -= 1;
	}

	// normalize
	dInputs = divide(dInputs, samples);
}

Optimizer_SGD::Optimizer_SGD(double learning_rate) {
	this->learning_rate = learning_rate;
}

void Optimizer_SGD::update(LayerDense* layer) {
	layer->weights = add(layer->weights, multiply(layer->weights, -1 * learning_rate));
	layer->biases = add(layer->biases, multiply(layer->biases, -1 * learning_rate));
}