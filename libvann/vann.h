#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <string>
#include <cstdarg>

#define DEFAULT_LEARNING_RATE 0.004
#define DEFAULT_LAMBDA 0.1
#define DEFAULT_MOMENTUM 0.0025
#define DEFAULT_MINI_BATCH 10

typedef double vann_type;

enum vann_activation_type {
	NO_FUNCTION,
	SIGMOID_ACTIVATION
};

enum vann_output_type {
	OUTPUT_NO_FUNCTION,
	OUTPUT_SIGMOID_MAX_NORMALIZATION,
	OUTPUT_SIGMOID_NORMALIZATION
};

class vann_data {
	unsigned int input_size, output_size;
	std::vector<vann_type*> v_input,  v_output;
public:
	unsigned int size();
	void addTest(vann_type *input, vann_type* output);
	void setTest(unsigned int test, vann_type *input, vann_type* output);
	vann_type* getInput(unsigned int test);
	vann_type* getOutput(unsigned int test);
	void clear();
	void set_data_from_file(char *file_name);
};

class vann {
	unsigned int *layers, num_layers, num_connections, num_neurons, max_layer_size;
	vann_type *weights, *weights_delta, *weights_batch_derivative, *biases, **derivative, *output, *biases_batch_derivative, *aux;
	bool usesMemory = false, use_custom_seed = false;
	vann_type WEIGHTS_DECAY = DEFAULT_LAMBDA, LEARNING_RATE = DEFAULT_LEARNING_RATE, MOMENTUM = DEFAULT_MOMENTUM, MINI_BATCH = DEFAULT_MINI_BATCH;
	vann_activation_type network_activation_type = SIGMOID_ACTIVATION;
	vann_output_type network_output_type = OUTPUT_SIGMOID_MAX_NORMALIZATION;
	unsigned int custom_seed = 0;

public:
	static void assert_network();

	vann_type MIN_WEIGTH_VALUE = -0.5, MAX_WEIGTH_VALUE = 0.5, MIN_BIAS_VALUE = -0.5, MAX_BIAS_VALUE = 0.5;

	vann_type randomWeight();

	vann_type randomBias();

	void vann::compute_output_function(vann_type *user_output, vann_type *output, unsigned int layer_size);

	void addAndCompute(double *output, double *biases, unsigned int layer_size);

	void vann::multiply(vann_type *dest, vann_type *output, vann_type *weights, unsigned int previous_layer, unsigned int next_layer);

	vann_type activation_function(vann_type value);

	vann_type activation_function_derivative(vann_type value);

	void setDecay(vann_type value);
	vann_type getDecay();
	void setLearningRate(vann_type value);
	vann_type getLearningRate();
	void setMomentum(vann_type value);
	vann_type getMomentum();
	void setMiniBatch(vann_type value);
	vann_type getMiniBatch();
	void setHiddenLayerActivationFunction(vann_activation_type activation_type);
	void setOutputLayerFunction(vann_output_type output_type);
	void setWeights(vann_type *weights);
	void setBiases(vann_type *biases);
	void setWeightsRange(vann_type minimum_weight_value, vann_type maximum_weight_value);
	void setBiasesRange(vann_type minimum_bias_value, vann_type maximum_bias_value);
	void setCustomSeed(unsigned int seed);

	void setLayers(const unsigned int num_layers, ...);
	void setLayers(const unsigned int num_layers, unsigned int *user_layers);
	void setLayersWithoutInitialising(const unsigned int num_layers, unsigned int *user_layers);
	
	void destroy();
	vann_type vann::get_mse(vann_data *data);

	

	vann_type error_function_derivative(vann_type output, vann_type expectedOutput);
	void resetBatchDerivatives();

	vann_type* feedforward(vann_type *input);
	void backpropagate(vann_type *expectedOutput);
	void gradient();



	vann_type learn_epoch(vann_data *data);
	bool learn_from_data(vann_data *data, unsigned int num_epoch, vann_type desired_error, unsigned int log_rate);
	bool learn_from_data(vann_data *data, unsigned int num_epoch, vann_type desired_error, unsigned int log_rate, unsigned int save_rate, char *file_name);
	bool vann::learn_from_data(vann_data *data, unsigned int num_epoch, vann_type desired_error, unsigned int log_rate, unsigned int save_rate, char *file_name, unsigned int validation_rate, vann_data *validation_data);

	char* getBytes();
	bool loadFromFile(char *file);
	void save_bytes_to_file(char *file_name);
};
