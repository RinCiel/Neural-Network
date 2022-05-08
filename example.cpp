#include "neural_network\include\neural_network.h"

#include "mnist\mnist_reader.hpp"

int main() {

    NeuralNetworkInit();

	// mnist tests
	MNIST_Reader mnist;
	mnist.readTrainingData();
	// display second number in training set
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (mnist.training_images[1].pixels[i * 28 + j] == 0) {
				std::cout << " ";
			}
			else {
				std::cout << "*";
			}
		}
		std::cout << std::endl;
	}
	//

    std::vector<std::vector<double>> inputs = {
		{0.7, 0.2, 0.1},
		{0.5, 0.1, 0.4},
		{0.02, 0.9, 0.08}
	};

	std::vector<double> targetOutput = { 0, 1, 1 };

	Layer_Dense layer(3, 5);
	Activation_ReLU relu;
	Layer_Dense layer2(5, 5);
	Activation_ReLU relu2;
	Layer_Dense layer3(5, 5);
	Activation_Softmax_Loss_CategoricalCrossEntropy activation_loss;

	layer.forward(inputs, true);
	relu.forward(layer.output);
	layer2.forward(relu.output);
	relu2.forward(layer2.output);
	layer3.forward(relu2.output);
	activation_loss.forward(layer3.output, targetOutput);

	Accuracy accuracy;
	accuracy.forward(activation_loss.softmax->output, targetOutput);

	std::cout << "Loss: " << activation_loss.loss->output << std::endl << "Accuracy: " << accuracy.output;

	// backpropagation
	activation_loss.backward(activation_loss.output, targetOutput);
	layer3.backward(activation_loss.dInputs);
	relu2.backward(layer3.dInputs);
	layer2.backward(relu2.dInputs);
	relu.backward(layer2.dInputs);
	layer.backward(relu.dInputs);

	// print out contents of layer dWeights
	std::cout << std::endl;
	print(layer.dWeights);
	std::cout << std::endl;
	print(layer.dBiases);

	// optimizer
	Optimizer_SGD optimizer;
	optimizer.update(&layer);
	optimizer.update(&layer2);
	optimizer.update(&layer3);
    return 0;
}