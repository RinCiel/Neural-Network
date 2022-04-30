#include "NeuralNetwork.h"

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
	Activation_Softmax softmax;

	layer.forward(inputs);
	relu.forward(layer.output);
	layer2.forward(relu.output);
	relu2.forward(layer2.output);
	layer3.forward(relu2.output);
	softmax.forward(layer3.output);

	Loss_CategoricalCrossEntropy loss;
	loss.forward(softmax.output, targetOutput);
	Accuracy accuracy;
	accuracy.forward(softmax.output, targetOutput);

	std::cout << "Loss: " << loss.output << std::endl << "Accuracy: " << accuracy.output;
    return 0;
}