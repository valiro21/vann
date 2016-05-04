#include <iostream>
#include "vann"

#define LEARNING_RATE 0.2
#define DECAY 0.00001
#define MOMENTUM 0.01

int main() {
	vann xor;
	vann_data xor_data;
	xor_data.set_data_from_file("xor.data");
	//xor.setCustomSeed(1454948552);
	xor.setLayers(3, 2, 2, 1);
	xor.setHiddenLayerActivationFunction(SIGMOID_ACTIVATION);
	xor.setOutputLayerFunction(OUTPUT_SIGMOID_NORMALIZATION);
	xor.setMiniBatch(4);
	xor.setLearningRate(LEARNING_RATE);
	xor.setDecay(DECAY);
	xor.setMomentum(MOMENTUM);
	/*double weights[] = { 1, 0.95, 1, 1, 1, 1 };
	double biases[] = {0, 0, 1, 1, 1};
	xor.setWeights(weights);
	xor.setBiases(biases);*/
	xor.learn_from_data(&xor_data, 10000000, 0.000001, 4000, 4000, "xor.net", 8000, &xor_data);
	xor.destroy();
	return 0;
}