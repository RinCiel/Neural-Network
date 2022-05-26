#include "neural_network\include\neural_network.h"

#include "mnist\mnist_reader.hpp"

#include <cstdio>

int main() {

    NeuralNetworkInit();

	MNIST_Reader mnist;
	mnist.readTrainingData();
	mnist.readTestData();
	mnist.randomTrainingData(500);

	// convert mnist.randomImages to inputs
	std::vector<std::vector<double>> inputs;
	for (int i = 0; i < mnist.randomImages.size(); i++) {
		std::vector<double> input;
		for (int j = 0; j < 784; j++) {
			input.push_back(mnist.randomImages[i].pixels[j]);
		}
		inputs.push_back(input);
	}

	// convert mnist.randomLabels to targetOutput
	std::vector<double> targetOutput;
	for (int i = 0; i < mnist.randomLabels.size(); i++) {
		targetOutput.push_back(mnist.randomLabels[i]);
	}

	Layer_Dense layer(784, 16);
	Activation_ReLU relu;
	Layer_Dense layer2(16, 16);
	Activation_ReLU relu2;
	Layer_Dense layer3(16, 10);
	Activation_Softmax_Loss_CategoricalCrossEntropy activation_loss;
	Accuracy accuracy;
	Optimizer_Adam optimizer(0.0005, 5e-7, 1e-7, 0.9, 0.999);

	for (int i = 0; i < 2000; i++) {
		layer.forward(inputs, true);
		relu.forward(layer.output);
		layer2.forward(relu.output);
		relu2.forward(layer2.output);
		layer3.forward(relu2.output);
		activation_loss.forward(layer3.output, targetOutput);

		accuracy.forward(activation_loss.softmax->output, targetOutput);
		std::cout << "Epoch: " << i + 1 << "|" << "Loss: " << activation_loss.loss->output << "|" << "Accuracy: " << accuracy.output << std::endl;
		
		// backpropagation
		activation_loss.backward(activation_loss.output, targetOutput);
		layer3.backward(activation_loss.dInputs);
		relu2.backward(layer3.dInputs);
		layer2.backward(relu2.dInputs);
		relu.backward(layer2.dInputs);
		layer.backward(relu.dInputs);

		// optimizer
		optimizer.applyDecay_pre();
		optimizer.update(&layer);
		optimizer.update(&layer2);
		optimizer.update(&layer3);
		optimizer.applyDecay_post();
	}

	mnist.randomTestData(100);

	// convert mnist.randomImages to inputs
	for (int i = 0; i < mnist.randomImages.size(); i++) {
		std::vector<double> input;
		for (int j = 0; j < 784; j++) {
			input.push_back(mnist.randomImages[i].pixels[j]);
		}
		inputs.push_back(input);
	}

	// convert mnist.randomLabels to targetOutput
	for (int i = 0; i < mnist.randomLabels.size(); i++) {
		targetOutput.push_back(mnist.randomLabels[i]);
	}

	layer.forward(inputs, true);
	relu.forward(layer.output);
	layer2.forward(relu.output);
	relu2.forward(layer2.output);
	layer3.forward(relu2.output);
	activation_loss.forward(layer3.output, targetOutput);
	
	accuracy.forward(activation_loss.softmax->output, targetOutput);
	std::cout << "Loss: " << activation_loss.loss->output << "|" << "Accuracy: " << accuracy.output << std::endl;

	std::getchar();
    return 0;
}