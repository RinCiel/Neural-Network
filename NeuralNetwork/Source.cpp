#include <iostream>
#include <vector>
#include <string.h>
#include "NeuralNetwork.h"

int main(){
	NeuralNetworkInit();
	// create input variables
	// using doubles as they are more precise than floats
	std::vector<std::vector<double>> inputs = { 
		{ -1 + (double)rand() / RAND_MAX * 2, -1 + (double)rand() / RAND_MAX * 2, -1 + (double)rand() / RAND_MAX * 2},
		{ -1 + (double)rand() / RAND_MAX * 2, -1 + (double)rand() / RAND_MAX * 2, -1 + (double)rand() / RAND_MAX * 2},
		{ -1 + (double)rand() / RAND_MAX * 2, -1 + (double)rand() / RAND_MAX * 2, -1 + (double)rand() / RAND_MAX * 2}
	};

	for (int i = 0; i < inputs.size(); i++)
	{
		std::cout << "[";
		for (int j = 0; j < inputs[i].size(); j++)
		{
			std::cout << inputs[i][j];
			if (j + 1 != inputs[i].size()) {
				std::cout << ", ";
			}
		}
		std::cout << "]";
		std::cout << std::endl;
	}

	LayerDense layer(3, 5);
	ActivationReLU relu;
	LayerDense layer2(5, 6);
	ActivationReLU relu2;
	LayerDense layer3(6, 10);
	ActivationReLU relu3;
	LayerDense layer4(10, 5);
	ActivationReLU relu4;
	ActivationSoftmax softmax;

	layer.forward(inputs);
	relu.forward(layer.outputs);
	layer2.forward(relu.outputs);
	relu2.forward(layer2.outputs);
	layer3.forward(relu2.outputs);
	relu3.forward(layer3.outputs);
	layer4.forward(relu3.outputs);
	relu4.forward(layer4.outputs);
	softmax.forward(relu4.outputs);
	softmax.displayOutputs();
}
