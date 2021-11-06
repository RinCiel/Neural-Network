#include <iostream>
#include <vector>
#include <string.h>
#include "NeuralNetwork.h"

void test() {
	// create input variables
	// using doubles as they are more precise than floats
	std::vector<std::vector<double>> inputs = {
		{0.7, 0.1, 0.2},
		{0.1, 0.5, 0.4},
		{0.02, 0.9, 0.08}
	};

	std::vector<std::vector<double>> targetOutput = {
		{1, 0, 0},
		{0, 1, 0},
		{0, 1, 0}
	};

	LayerDense layer(3, 5);
	ActivationReLU relu;
	LayerDense layer2(5, 5);
	ActivationReLU relu2;
	LayerDense layer3(5, 5);
	ActivationSoftmax softmax;

	layer.forward(inputs);
	relu.forward(layer.output);
	layer2.forward(relu.output);
	relu2.forward(layer2.output);
	layer3.forward(relu2.output);
	softmax.forward(layer3.output);

	LossCategoricalCrossEntropy loss;
	loss.forward(softmax.output, targetOutput);
	std::cout << "Loss: " << loss.output;
}

int main() {
	NeuralNetworkInit();
	test();
}
