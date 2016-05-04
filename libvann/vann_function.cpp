#include "vann.h"

vann_type vann::randomWeight() {
	return MIN_WEIGTH_VALUE + ((vann_type)rand() / (vann_type)RAND_MAX) * (MAX_WEIGTH_VALUE - MIN_WEIGTH_VALUE);
}

vann_type vann::randomBias() {
	return MIN_BIAS_VALUE + ((vann_type)rand() / (vann_type)RAND_MAX) * (MAX_BIAS_VALUE - MIN_BIAS_VALUE);
}

vann_type vann::activation_function(vann_type value) {
	switch (network_activation_type) {
	case NO_FUNCTION:
		return value;
	case SIGMOID_ACTIVATION:
		return 1. / (1. + exp(-value));
	}
}

vann_type vann::activation_function_derivative(vann_type value) {
	switch (network_activation_type) {
	case NO_FUNCTION:
		return 1;
	case SIGMOID_ACTIVATION:
		vann_type dvalue = activation_function(value);
		return dvalue * (1. - dvalue);
	}
}

vann_type vann::error_function_derivative(vann_type output, vann_type expectedOutput) {
	return (output - expectedOutput) * activation_function_derivative(output);
}

void vann::multiply(vann_type *dest, vann_type *output, vann_type *weights, unsigned int previous_layer, unsigned int next_layer) {
	for (int j = 0; j < next_layer; j++) {
		dest[j] = 0;
		for (int i = 0; i < previous_layer; i++)
			dest[j] += weights[i * next_layer + j] * output[i];
	}
}

vann_type pw2(vann_type value) {
	return value * value;
}

vann_type vann::get_mse(vann_data *data) {
	vann_type mse = 0;
	for (unsigned int t = 0; t < data->size(); t++) {
		feedforward(data->getInput(t));
		for (unsigned int i = 0; i < layers[num_layers - 1]; i++)
			mse += pw2((output[num_neurons - layers[num_layers - 1] + i] - data->getOutput(t)[i]));
	}

	mse /= (vann_type)data->size();
	return mse;
}

void vann::addAndCompute(double *output, double *biases, unsigned int layer_size) {
	for (int i = 0; i < layer_size; i++)
		output[i] = activation_function(output[i] + biases[i]);
}

void vann::compute_output_function(vann_type *user_output, vann_type *output, unsigned int layer_size) {
	vann_type max = output[0];
	unsigned int max_index = 0;

	switch (network_output_type) {
	case OUTPUT_NO_FUNCTION:
		memcpy(user_output, output, layer_size * sizeof(vann_type));
		break;
	case OUTPUT_SIGMOID_MAX_NORMALIZATION:
		
		for (unsigned int i = 1; i < layer_size; i++)
			if (output[i] > max)
				max = output[i],
				max_index = i;
		memset(user_output, 0, layer_size * sizeof(vann_type));
		user_output[max_index] = 1;
		break;
	case OUTPUT_SIGMOID_NORMALIZATION:
		for (unsigned int i = 0; i < layer_size; i++) {
			if (output[i] >= 0.5)
				user_output[i] = 1;
			else
				user_output[i] = 0;
		}
		break;
	}
}