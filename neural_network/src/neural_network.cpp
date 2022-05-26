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

Layer_Dense::Layer_Dense(int inputs, int neurons, double weight_reg_L1, double weight_reg_L2, double bias_reg_L1, double bias_reg_L2) {
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

	// regularizer
	this->weight_reg_L1 = weight_reg_L1;
	this->weight_reg_L2 = weight_reg_L2;
	this->bias_reg_L1 = bias_reg_L1;
	this->bias_reg_L2 = bias_reg_L2;
}

void Layer_Dense::forward(std::vector<std::vector<double>> inputs, bool normalize) {
	input = inputs;
	if (normalize) {
		// normalize inputs
		for (int i = 0; i < input.size(); i++) {
			double sum = getSum(input[i]);
			for (int j = 0; j < input[i].size(); j++) {
				input[i][j] /= sum;
			}
		}
	}
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
void Layer_Dense::backward(std::vector<std::vector<double>> inputs) {
	dWeights = dot(transpose(input), inputs);
	dBiases = sumVertical(inputs);

	if (this->weight_reg_L1 > 0.0) {
		std::vector<std::vector<double>> sign_weights = sign(weights);
		this->dWeights = add(this->dWeights, multiply(sign_weights, this->weight_reg_L1));
	}

	if (this->weight_reg_L2 > 0.0) {
		// dWeights += 2 * weight_reg_l2 * weights
		this->dWeights = add(this->dWeights, multiply(multiply(this->weights, this->weight_reg_L2), 2));
	}

	if (this->bias_reg_L1 > 0.0) {
		std::vector<double> sign_biases = sign(biases);
		this->dBiases = add(this->dBiases, multiply(sign_biases, this->bias_reg_L1));
	}

	if (this->bias_reg_L2 > 0.0) {
		// dBiases += 2 * bias_reg_l2 * biases
		this->dBiases = add(this->dBiases, multiply(multiply(this->biases, this->bias_reg_L2), 2));
	}

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

Optimizer_SGD::Optimizer_SGD(double learning_rate, double decay, double momentum) {
	iterations = 0;
	this->learning_rate = learning_rate;
	this->current_learning_rate = learning_rate;
	this->decay = decay;
	this->momentum = momentum;
}

void Optimizer_SGD::update(Layer_Dense* layer) {
	if (momentum > 0.0) {
		applyMomentum(layer);
	}
	else {
		layer->weights = add(layer->weights, multiply(layer->dWeights, -1 * learning_rate));
		layer->biases = add(layer->biases, multiply(layer->dBiases, -1 * learning_rate));
	}
}

void Optimizer_SGD::applyDecay_pre() {
	if (decay > 0.0) {
		current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
	}
}

void Optimizer_SGD::applyDecay_post() {
	iterations += 1;
}

void Optimizer_SGD::applyMomentum(Layer_Dense* layer) {
	if (!layer->momentum_set) {
		layer->momentum_weights = std::vector<std::vector<double>>(layer->weights.size(), std::vector<double>(layer->weights[0].size(), 0));
		layer->momentum_biases = std::vector<double>(layer->biases.size(), 0);
		layer->momentum_set = true;
	}
	layer->momentum_weights = add(multiply(layer->momentum_weights, momentum), multiply(layer->dWeights, -1 * current_learning_rate));
	layer->momentum_biases = add(multiply(layer->momentum_biases, momentum), multiply(layer->dBiases, -1 * current_learning_rate));
	layer->weights = add(layer->weights, layer->momentum_weights);
	layer->biases = add(layer->biases, layer->momentum_biases);
}

Optimizer_Adagrad::Optimizer_Adagrad(double learning_rate, double decay, double epsilon) {
	this->iterations = 0;
	this->learning_rate = learning_rate;
	this->current_learning_rate = learning_rate;
	this->decay = decay;
	this->epsilon = epsilon;
}

void Optimizer_Adagrad::update(Layer_Dense* layer) {
	if (!layer->cache_set) {
		layer->cache_weights = std::vector<std::vector<double>>(layer->weights.size(), std::vector<double>(layer->weights[0].size(), 0));
		layer->cache_biases = std::vector<double>(layer->biases.size(), 0);
		layer->cache_set = true;
	}
	layer->cache_weights = add(layer->cache_weights, multiply(layer->dWeights, layer->dWeights));
	layer->cache_biases = add(layer->cache_biases, multiply(layer->dBiases, layer->dBiases));
	
	layer->weights = add(layer->weights, divide(multiply(layer->dWeights, -1 * current_learning_rate), add(sqrt(layer->cache_weights), epsilon)));
	layer->biases = add(layer->biases, divide(multiply(layer->dBiases, -1 * current_learning_rate), add(sqrt(layer->cache_biases), epsilon)));
}

void Optimizer_Adagrad::applyDecay_pre() {
	if (decay > 0.0) {
		current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
	}
}

void Optimizer_Adagrad::applyDecay_post() {
	iterations += 1;
}

Optimizer_RMSProp::Optimizer_RMSProp(double learning_rate, double decay, double epsilon, double rho) {
	this->iterations = 0;
	this->learning_rate = learning_rate;
	this->current_learning_rate = learning_rate;
	this->decay = decay;
	this->epsilon = epsilon;
	this->rho = rho;
}

void Optimizer_RMSProp::update(Layer_Dense* layer) {
	if (!layer->cache_set) {
		layer->cache_weights = std::vector<std::vector<double>>(layer->weights.size(), std::vector<double>(layer->weights[0].size(), 0));
		layer->cache_biases = std::vector<double>(layer->biases.size(), 0);
		layer->cache_set = true;
	}
	// cache_weights = rho * weight_cache + (1 - rho) * weight_delta ** 2
	layer->cache_weights = add(multiply(layer->cache_weights, rho), multiply(multiply(layer->dWeights, layer->dWeights), 1 - rho));
	// cache_biases = rho * bias_cache + (1 - rho) * bias_delta ** 2
	layer->cache_biases = add(multiply(layer->cache_biases, rho), multiply(multiply(layer->dBiases, layer->dBiases), 1 - rho));

	// layer->weights += -1 * current_learning_rate * layer.dWeights / (sqrt(layer.cache_weights) + epsilon)
	layer->weights = add(layer->weights, divide(multiply(layer->dWeights, -1 * current_learning_rate), add(sqrt(layer->cache_weights), epsilon)));
	// layer->biases += -1 * current_learning_rate * layer.dBiases / (sqrt(layer.cache_biases) + epsilon)
	layer->biases = add(layer->biases, divide(multiply(layer->dBiases, -1 * current_learning_rate), add(sqrt(layer->cache_biases), epsilon)));
}

void Optimizer_RMSProp::applyDecay_pre() {
	if (decay > 0.0) {
		current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
	}
}

void Optimizer_RMSProp::applyDecay_post() {
	iterations += 1;
}

Optimizer_Adam::Optimizer_Adam(double learning_rate, double decay, double epsilon, double beta_1, double beta_2) {
	this->iterations = 0;
	this->learning_rate = learning_rate;
	this->current_learning_rate = learning_rate;
	this->decay = decay;
	this->epsilon = epsilon;
	this->beta_1 = beta_1;
	this->beta_2 = beta_2;
}

void Optimizer_Adam::update(Layer_Dense* layer) {
	if (!layer->cache_set) {
		layer->momentum_weights = std::vector<std::vector<double>>(layer->weights.size(), std::vector<double>(layer->weights[0].size(), 0));
		layer->cache_weights = std::vector<std::vector<double>>(layer->weights.size(), std::vector<double>(layer->weights[0].size(), 0));

		layer->momentum_biases = std::vector<double>(layer->biases.size(), 0);
		layer->cache_biases = std::vector<double>(layer->biases.size(), 0);
	}
	
	// layer.momentum_weights = beta_1 * layer.momentum_weights + (1 - beta_1) * layer.dWeights
	layer->momentum_weights = add(multiply(layer->momentum_weights, beta_1), multiply(layer->dWeights, 1 - beta_1));
	// layer.momentum_biases = beta_1 * layer.momentum_biases + (1 - beta_1) * layer.dBiases
	layer->momentum_biases = add(multiply(layer->momentum_biases, beta_1), multiply(layer->dBiases, 1 - beta_1));

	// correct momentums
	std::vector<std::vector<double>> momentum_weights_corrected = divide(layer->momentum_weights, 1 - pow(beta_1, iterations + 1));
	std::vector<double> momentum_biases_corrected = divide(layer->momentum_biases, 1 - pow(beta_1, iterations + 1));

	// layer.cache_weights = beta_2 * layer.cache_weights + (1 - beta_2) * layer.dWeights ** 2
	layer->cache_weights = add(multiply(layer->cache_weights, beta_2), multiply(multiply(layer->dWeights, layer->dWeights), 1 - beta_2));
	// layer.cache_biases = beta_2 * layer.cache_biases + (1 - beta_2) * layer.dBiases ** 2
	layer->cache_biases = add(multiply(layer->cache_biases, beta_2), multiply(multiply(layer->dBiases, layer->dBiases), 1 - beta_2));

	// correct caches
	std::vector<std::vector<double>> cache_weights_corrected = divide(layer->cache_weights, 1 - pow(beta_2, iterations + 1));
	std::vector<double> cache_biases_corrected = divide(layer->cache_biases, 1 - pow(beta_2, iterations + 1));

	// layer.weights += -1 * current_learning_rate * momentum_weights_corrected / (sqrt(cache_weights_corrected) + epsilon)
	layer->weights = add(layer->weights, divide(multiply(momentum_weights_corrected, -1 * current_learning_rate), add(sqrt(cache_weights_corrected), epsilon)));

	// layer.biases += -1 * current_learning_rate * momentum_biases_corrected / (sqrt(cache_biases_corrected) + epsilon)
	layer->biases = add(layer->biases, divide(multiply(momentum_biases_corrected, -1 * current_learning_rate), add(sqrt(cache_biases_corrected), epsilon)));
}

void Optimizer_Adam::applyDecay_pre() {
	if (decay > 0.0) {
		current_learning_rate = learning_rate * (1.0 / (1.0 + decay * iterations));
	}
}

void Optimizer_Adam::applyDecay_post() {
	iterations += 1;
}

double regularization_loss(Layer_Dense* layer) {
	double loss = 0;
	
	if (layer->weight_reg_L1 > 0.0) {
		for (int i = 0; i < layer->weights.size(); i++) {
			for (int j = 0; j < layer->weights[i].size(); j++) {
				loss += layer->weight_reg_L1 * fabs(layer->weights[i][j]);
			}
		}
	}

	if (layer->weight_reg_L2 > 0.0) {
		for (int i = 0; i < layer->weights.size(); i++) {
			for (int j = 0; j < layer->weights[i].size(); j++) {
				loss += layer->weight_reg_L2 * layer->weights[i][j] * layer->weights[i][j];
			}
		}
	}

	if (layer->bias_reg_L1 > 0.0) {
		for (int i = 0; i < layer->biases.size(); i++) {
			loss += layer->bias_reg_L1 * fabs(layer->biases[i]);
		}
	}

	if (layer->bias_reg_L2 > 0.0) {
		for (int i = 0; i < layer->biases.size(); i++) {
			loss += layer->bias_reg_L2 * layer->biases[i] * layer->biases[i];
		}
	}

	return loss;
}