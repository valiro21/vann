#include "vann.h"
#include <assert.h>

void vann::assert_network() {
	//testing matrix multiplication and vectors add
	vann_type m[9] = {
		1., 0., 0.,
		0., 1., 0.,
		0., 0., 1.
	}, v[3] = { 1., 2., 3. }, v2[3], v3[3] = {6, 5, 4};

	vann k;
	k.multiply(v2, v, m, 3, 3);
	for (int i = 0; i < 3; i++)
		assert(v2[i] == i + 1);

	k.setHiddenLayerActivationFunction(NO_FUNCTION);
	k.addAndCompute(v2, v3, 3);

	for (int i = 0; i < 3; i++)
		assert(v2[i] == 7);

	k.multiply(v, v2, m, 3, 3);
	for (int i = 0; i < 3; i++)
		assert(v[i] == 7);

	k.addAndCompute(v, v3, 3);
	for (int i = 0; i < 3; i++)
		assert(v[i] == 13 - i);


	//testing save and load functions
	k.setLayers(3, 2, 3, 4);
	k.save_bytes_to_file("test.net");
	vann k2;
	k2.loadFromFile("test.net");

	int temp = memcmp(k.weights, k2.weights, k.num_connections);
	assert(k.num_layers == k2.num_layers);
	assert(k.num_neurons == k2.num_neurons);
	assert(k.num_connections == k2.num_connections);
	assert(memcmp(k.layers, k2.layers, k.num_layers * 4) == 0);
	assert(memcmp(k.biases, k2.biases, k.num_neurons * 8) == 0);
	assert(memcmp(k.weights, k2.weights, k.num_connections * 8) == 0);

	//verify feedforward method
	vann feed;
	feed.setHiddenLayerActivationFunction(SIGMOID_ACTIVATION);
	feed.setLayers(3, 2, 1, 2);
	double biases[] = {0, 0, 1.4, 0, 0};
	double weights[] = { 0.56, -0.22, 1.5, -4};
	memcpy(feed.biases, biases, 40);
	memcpy(feed.weights, weights, 32);
	double input[] = { 8, 3 };
	feed.feedforward(input);
	double output[] = { 8., 3., 0.99462175282615140, 0.81636817247023175, 0.018370154084120776 };
	assert(memcmp(feed.output, output, 40) == 0);

	//verify backpropagation method
	double desiredOutput[] = {1, 0};
	feed.resetBatchDerivatives();
	feed.backpropagate(desiredOutput);
	double biases_derivatives[] = { -0.015161115510944963, -0.039034865064649663, 0.0045921510903681505};
	assert (memcmp(feed.biases_batch_derivative+2, biases_derivatives, 24) == 0);
	double weights_derivatives[] = { -0.12128892408755970, -0.045483346532834888, -0.038824925911934149, 0.0045674533667444924};
	assert(memcmp(feed.weights_batch_derivative, weights_derivatives, 32) == 0);

	//verify gradient method
	feed.setLearningRate(0.001);
	feed.setDecay(0.2);
	feed.setMomentum(0.4);
	double new_weights[] = { 0.5597667110759124403, -0.219910516653467165112, 1.500338824925911934149, -3.9992045674533667444924 };
	double new_biases[] = {1.4000151611155109, 0.000039034865064649667, -0.0000045921510903681505};
	feed.gradient();
	assert(memcmp(feed.biases + 2, new_biases, 24) == 0);
	assert(memcmp(feed.weights, new_weights, 32));

	k.destroy();
	k2.destroy();
	feed.destroy();
}