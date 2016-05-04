#include "vann.h"

void vann::setDecay(vann_type value) {
	WEIGHTS_DECAY = value;
}

vann_type vann::getDecay() {
	return WEIGHTS_DECAY;
}

void vann::setLearningRate(vann_type value) {
	LEARNING_RATE = value;
}

vann_type vann::getLearningRate() {
	return LEARNING_RATE;
}

void vann::setMomentum(vann_type value) {
	MOMENTUM = value;
}

void vann::setWeights(vann_type *weights) {
	memcpy(this->weights, weights, num_connections * sizeof(vann_type));
}

void vann::setBiases(vann_type *biases) {
	memcpy(this->biases, biases, num_neurons * sizeof(vann_type));
}

void vann::setWeightsRange(vann_type minimum_weight_value, vann_type maximum_weight_value) {
	this->MIN_WEIGTH_VALUE = minimum_weight_value;
	this->MAX_WEIGTH_VALUE = maximum_weight_value;
}

void vann::setBiasesRange(vann_type minimum_bias_value, vann_type maximum_bias_value) {
	this->MIN_BIAS_VALUE = minimum_bias_value;
	this->MAX_BIAS_VALUE = maximum_bias_value;
}

void vann::setCustomSeed(unsigned int seed) {
	use_custom_seed = true;
	custom_seed = seed;
}

vann_type vann::getMomentum() {
	return MOMENTUM;
}

void vann::setMiniBatch(vann_type value) {
	MINI_BATCH = value;
}

vann_type vann::getMiniBatch() {
	return MINI_BATCH;
}


void vann::setHiddenLayerActivationFunction(vann_activation_type activation_type) {
	network_activation_type = activation_type;
}

void vann::setOutputLayerFunction(vann_output_type output_type) {
	network_output_type = output_type;
}