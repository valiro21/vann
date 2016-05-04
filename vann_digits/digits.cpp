#include <iostream>
#include "vann"

#define LEARNING_RATE 0.2
#define DECAY 0
#define MOMENTUM 0.15

int main() {
	vann xor;
	vann_data digits_data, digits_validation;
	digits_data.set_data_from_file("digits.data");
	digits_validation.set_data_from_file("validation_digits.data");
	xor.loadFromFile("digits.net");
	//xor.setLayers(3, 28*28, 300, 10);
	xor.setHiddenLayerActivationFunction(SIGMOID_ACTIVATION);
	xor.setOutputLayerFunction(OUTPUT_SIGMOID_MAX_NORMALIZATION);
	xor.setMiniBatch(10);
	xor.setLearningRate(LEARNING_RATE);
	xor.setDecay(DECAY);
	xor.setMomentum(MOMENTUM);
	xor.learn_from_data(&digits_data, 10000000, 0.000001, 50, 50, "digits.net", 100, &digits_validation);
	xor.destroy();
	return 0;
}