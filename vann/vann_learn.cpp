#include "vann.h"
#include <algorithm>

bool vann::learn_from_data(vann_data *data, unsigned int num_epoch, vann_type desired_error, unsigned int log_rate) {
	return learn_from_data(data, num_epoch, desired_error, log_rate, 1, NULL);
}

bool vann::learn_from_data(vann_data *data, unsigned int num_epoch, vann_type desired_error, unsigned int log_rate, unsigned int save_rate, char *file_name) {
	return learn_from_data(data, num_epoch, desired_error, log_rate, 1, NULL, 1, NULL);
}

bool vann::learn_from_data(vann_data *data, unsigned int num_epoch, vann_type desired_error, unsigned int log_rate, unsigned int save_rate, char *file_name, unsigned int validation_rate, vann_data *validation_data) {
	std::vector<int> mb;
	for (int i = 0; i < data->size(); i++)
		mb.push_back(i);

	vann_data *batch_data = new vann_data();

	for (unsigned int e = 0; e <= num_epoch; e++) {
		std::random_shuffle(mb.begin(), mb.end());

		batch_data->clear();
		for (int i = 0; i < MINI_BATCH && i < data->size(); i++)
			batch_data->addTest(data->getInput(mb[i]), data->getOutput(mb[i]));

		vann_type error = learn_epoch(batch_data);

		if (error < desired_error) {
			if (file_name != NULL)
				save_bytes_to_file(file_name);
			return true;
		}

		if (e % log_rate == 0) {
			//print log
			printf("Epoch %ld:  error = %.15lf\n", e, error);
		}

		if (file_name != NULL && e % save_rate == 0) {
			//save network
			save_bytes_to_file(file_name);
		}

		if (validation_data != NULL && e % validation_rate == 0) {
			unsigned int success = 0;
			vann_type *aux = new vann_type[layers[num_layers-1]];
			for (int t = 0; t < validation_data->size(); t++) {
				double *output = feedforward(validation_data->getInput(t));
				compute_output_function(aux, output, layers[num_layers-1]);
				bool okay = true;
				for (int i = 0; i < layers[num_layers - 1]; i++)
					if (aux[i] != validation_data->getOutput(t)[i]) {
						okay = false;
						break;
					}
				success += okay;
			}

			printf(" %ld / %ld tests cleread\n", success, validation_data->size());
			delete[] aux;
		}
	}

	return false;
}

vann_type vann::learn_epoch(vann_data *data) {
	resetBatchDerivatives();
	for (int i = 0; i < data->size(); i++) {
		feedforward(data->getInput(i));
		backpropagate(data->getOutput(i));
	}
	gradient();
	return get_mse(data);
}

void vann::gradient() {
	unsigned int num_connections = 0, num_neurons = layers[0], weights_index;
	for (unsigned int l = 1; l < num_layers; l++) {
		for (unsigned int i = 0; i < layers[l]; i++) {
			biases[num_neurons + i] -= LEARNING_RATE * biases_batch_derivative[num_neurons + i];
			for (unsigned int j = 0; j < layers[l - 1]; j++) {
				weights_index = num_connections + j * layers[l] + i;
				weights_delta[weights_index] = LEARNING_RATE * weights_batch_derivative[weights_index] / MINI_BATCH + WEIGHTS_DECAY * LEARNING_RATE * weights[weights_index] - weights_delta[weights_index] * MOMENTUM;
				weights[weights_index] -= weights_delta[weights_index];
			}
		}

		num_connections += layers[l] * layers[l - 1];
		num_neurons += layers[l];
	}
}

void vann::backpropagate(vann_type *expectedOutput) {
	for (int i = 0; i < layers[num_layers - 1]; i++)
		derivative[0][i] = error_function_derivative(output[num_neurons - layers[num_layers - 1] + i], expectedOutput[i]),
		biases_batch_derivative[num_neurons - layers[num_layers - 1] + i] += derivative[0][i];

	bool okay = 1;
	double temp_derivative;
	unsigned int num_neurons = this->num_neurons - layers[num_layers - 1], num_connections = this->num_connections;
	for (int l = num_layers - 2; l >= 0; l--, okay = !okay) {
		memset(derivative[okay], 0, max_layer_size * sizeof(vann_type));
		num_connections -= layers[l] * layers[l + 1];
		for (int j = 0; j < layers[l]; j++) {
			for (int i = 0; i < layers[l + 1]; i++) {
				temp_derivative = weights[num_connections + j * layers[l + 1] + i] * derivative[!okay][i];
				derivative[okay][j] += temp_derivative;
				weights_batch_derivative[num_connections + j * layers[l+1] + i] += output[num_neurons - layers[l] + j] * derivative[!okay][i];
			}
			derivative[okay][j] *= activation_function_derivative(output[num_neurons - layers[l] + j]);

			biases_batch_derivative[num_neurons - layers[l] + j] += derivative[okay][j];
		}

		num_neurons -= layers[l];
	}
}

vann_type* vann::feedforward(vann_type *input) {
	unsigned int num_connections = 0;
	memcpy(output, input, layers[0] * sizeof(vann_type));

	unsigned int offset = 0;
	for (int i = 1; i < num_layers; i++) {
		multiply(output + offset + layers[i-1], output + offset, &(weights[num_connections]), layers[i-1], layers[i]);
		offset += layers[i-1];
		addAndCompute(output + offset, biases + offset, layers[i]);

		num_connections += layers[i - 1] * layers[1];
	}

	return output + offset;
}