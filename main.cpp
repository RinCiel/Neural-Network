#include "neural_network.h"

// print hello world
int main() {

    NeuralNetworkInit();
    std::vector<std::vector<double>> inputs = {
		{0.7, 0.2, 0.1},
		{0.5, 0.1, 0.4},
		{0.02, 0.9, 0.08}
	};

	std::vector<double> targetOutput = { 0, 1, 1 };

	LayerDense layer(3, 5);
	Activation_ReLU relu;
	LayerDense layer2(5, 5);
	Activation_ReLU relu2;
	LayerDense layer3(5, 5);
	Activation_Softmax_Loss_CategoricalCrossEntropy activation_loss;

	layer.forward(inputs);
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