/*
Library for Neural Networks

NOTES:
- NeuronNetWorkInit() must be called first
- Each class contains a 'forward' method that must be called to calculate the outputs
*/

#include "NeuralNetwork.h"

void NeuralNetworkInit() {
	// set random seed
	srand(time(NULL));
}

// ========================================================================================================================
// LayerDense class
// makes a neuron layer with inputs and any number of neurons
LayerDense::LayerDense(int inputsIn, int neuronsIn) {
	/*
	vectorInputs = data to be passed
	inputs = features per data set
	neurons = any number
	the neurons of a previous layer has to be the same as the inputs of the current layer
	*/
	inputs = inputsIn;
	neurons = neuronsIn;

	//randomly generate a 2d vector of [inputs][neurons] with a range of -1 to 1
	std::vector<std::vector<double>> weights; // create a 2d vector (all the data of the weights)
	std::vector<double> aNeuronWeights = {}; // create a neuron with an empty vector
	std::vector<std::vector<double>> output; // batch samples require 3d vectors

	for (int i = 0; i < neurons; i++) { // loop through each neuron in the layer
		for (int j = 0; j < inputs; j++) { // add values between -1 and 1 for each neuron
			aNeuronWeights.push_back(-1 + (double)rand() / RAND_MAX * 2);
		}
		weights.push_back(aNeuronWeights);
		aNeuronWeights = {};
	}
	layerWeights = weights;

	std::vector<double> biases;
	// set biases to 0 (it can be any number though)
	for (int k = 0; k < neurons; k++) {
		biases.push_back(0);
	}
	layerBiases = biases;
}

void LayerDense::forward(std::vector<std::vector<double>> vectorInputsIn) {
	vectorInputs = vectorInputsIn;

	std::vector<std::vector<double>> results;
	std::vector<double> aResult;
	double aNeuronResult = 0;
	std::vector<double> currentSample;
	for (int i = 0; i < vectorInputs.size(); i++) { // each sample of inputs
		for (int j = 0; j < neurons; j++) { // for each neuron
			bool biasAdded = false;
			for (int k = 0; k < inputs; k++) { // for each weight in neuron (same as sample inputs)
				aNeuronResult += layerWeights[j][k] * vectorInputs[i][k]; // multiply inputs with weights
				if (!biasAdded) { // add bias
					aNeuronResult += layerBiases[j];
					biasAdded = true;
				}
			}
			// finished with one neuron, push into aResult
			aResult.push_back(aNeuronResult);
			// clear the neuron result
			aNeuronResult = 0;
		}
		// push the current sample neuron results
		results.push_back(aResult);
		// clear current sample
		aResult = {};
	}
	// move our results into output
	output = results;
}

void LayerDense::displayWeights() {
	std::cout << "====================================" << std::endl;
	std::cout << "Weights" << std::endl;
	for (int i = 0; i < neurons; i++)
	{
		std::cout << "[";
		for (int j = 0; j < inputs; j++)
		{
			std::cout << layerWeights[i][j];
			if (j + 1 != inputs) {
				std::cout << ", ";
			}
		}
		std::cout << "]";
		std::cout << std::endl;
	}
}

void LayerDense::displayBiases() {
	std::cout << "====================================" << std::endl;
	std::cout << "Biases" << std::endl;
	std::cout << "[";
	for (int i = 0; i < layerBiases.size(); i++) {
		std::cout << layerBiases[i];
		if (i + 1 != layerBiases.size()) {
			std::cout << ", ";
		}
	}
	std::cout << "]" << std::endl;
}

void LayerDense::displayOutputs() {
	std::cout << "====================================" << std::endl;
	std::cout << "Outputs" << std::endl;
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << "[";
		for (int j = 0; j < neurons; j++)
		{
			std::cout << output[i][j];
			if (j + 1 != neurons) {
				std::cout << ", ";
			}
		}
		std::cout << "]";
		std::cout << std::endl;
	}
}
// ========================================================================================================================
/*
ReLU activation function
Used after passing values through layers to get non-linear data
*/

// change the results to 0 if the results in the vector is negative
void ActivationReLU::forward(std::vector<std::vector<double>> inputsIn) {
	output = inputsIn;
	for (int i = 0; i < output.size(); i++) {
		for (int j = 0; j < output[i].size(); j++) {
			if (output[i][j] < 0) {
				output[i][j] = 0;
			}
		}
	}
}

void ActivationReLU::displayOutputs() {
	std::cout << "====================================" << std::endl;
	std::cout << "Outputs" << std::endl;
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << "[";
		for (int j = 0; j < output[i].size(); j++)
		{
			std::cout << output[i][j];
			if (j + 1 != output[i].size()) {
				std::cout << ", ";
			}
		}
		std::cout << "]";
		std::cout << std::endl;
	}
}

// ========================================================================================================================
/*
Softmax activation function
Usually used in the last layer so that the values in the layer are normalized (is between 0 and 1)
*/

void ActivationSoftmax::forward(std::vector<std::vector<double>> inputsIn) {
	output = inputsIn;
	double max;
	std::vector<double> row; // a row of outputs after softmax activation
	std::vector<std::vector<double>> beforeNormalization;
	for (int i = 0; i < output.size(); i++) {
		max = getArgMax(output[i]);
		for (int j = 0; j < output[i].size(); j++) {
			row.push_back(exp(output[i][j] - max)); // exponentiate values after subtracting the value from the max (prevents overflow)
		}
		beforeNormalization.push_back(row);
		row = {};
	}
	output = beforeNormalization;

	// Normalize values by calculating average
	double sum;
	for (int i = 0; i < output.size(); i++) {
		sum = getSum(output[i]);
		for (int j = 0; j < output[i].size(); j++) {
			output[i][j] = output[i][j] / sum;
		}
	}
}

void ActivationSoftmax::displayOutputs() {
	std::cout << "====================================" << std::endl;
	std::cout << "Outputs" << std::endl;
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << "[";
		for (int j = 0; j < output[i].size(); j++)
		{
			std::cout << output[i][j];
			if (j + 1 != output[i].size()) {
				std::cout << ", ";
			}
		}
		std::cout << "]";
		std::cout << std::endl;
	}
}
// ========================================================================================================================
/*
Categorical Cross Entropy Loss function
takes the negative log of the target output
negative log from 0 to 1 is between inf to 0

Used after the inputs have been passed through so it can calulate how confident the result is
The greater the loss, the less confident
*/
void LossCategoricalCrossEntropy::forward(std::vector<std::vector<double>> inputs, std::vector<double> targets) {
	std::vector<double> allLoss;
	std::vector<double> current;
	for (int i = 0; i < inputs.size(); i++) {
		current = clip(1e-7, 1 - 1e-7, inputs[i]);
		// removes 0 from negative log and 1 to make it fairer

		// for each sample, get the negative log of the target confidence
		allLoss.push_back(-1 * log(current[targets[i]]));
		std::cout << current[targets[i]] << std::endl;
	}
	output = getSum(allLoss) / allLoss.size();
}

// overload the forward function to account for 2d vectors
void LossCategoricalCrossEntropy::forward(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
	std::vector<double> current;
	std::vector<double> allLoss;
	for (int i = 0; i < inputs.size(); i++) {
		current = clip(1e-7, 1 - 1e-7, inputs[i]);
		for (int j = 0; j < current.size(); j++) {
			// multiply current value with target value
			current[j] = current[j] * targets[i][j];
		}
		// since most of it is multiplied by 0, the sum is the target confidence
		allLoss.push_back(-1 * log(getSum(current)));
		current = {};
 	}
	output = getSum(allLoss) / allLoss.size();
}

// ========================================================================================================================
// Helper functions

// get largest value in the vector
double getArgMax(std::vector<double> vec) {
	double res = vec[0];
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] > res) {
			res = vec[i];
		}
	}
	return res;
}

// get the sum of the elements in the vector
double getSum(std::vector<double> vec) {
	double res = 0;
	for (int i = 0; i < vec.size(); i++) {
		res += vec[i];
	}
	return res;
}

std::vector<double> clip(double min, double max, std::vector<double> vec) {
	for (int i = 0; i < vec.size(); i++) {
		if (vec[i] < min) {
			vec[i] = min;
		}
		else if (vec[i] > max) {
			vec[i] = max;
		}
	}
	return vec;
}
