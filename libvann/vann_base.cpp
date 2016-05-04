#include "vann.h"
#include <stdio.h>

char* vann::getBytes() {
	unsigned int offset = 0, bytes_size = num_layers * 4 + num_neurons * 8 + num_connections * 8 + 17;
	char* bytes = new char[bytes_size + 1];

	memcpy(bytes, &bytes_size, 4);
	memcpy(bytes + 4, &num_layers, 4);
	memcpy(bytes + 8, layers, num_layers * 4);
	offset = 8 + num_layers * 4;

	memcpy(bytes + offset, &num_neurons, 4);
	memcpy(bytes + offset + 4, biases, num_neurons * 8);
	offset += 4 + num_neurons * 8;

	memcpy(bytes + offset, &num_connections, 4);
	memcpy(bytes + offset + 4, weights, 8 * num_connections);
	offset += 4 + num_connections * 8;

	memcpy(bytes + offset, "1", 1);
	return bytes;
}

void vann::save_bytes_to_file(char *file_name) {
	char *bytes = getBytes();
	unsigned int bytes_size;
	memcpy(&bytes_size, bytes, 4);

	std::ofstream fout(file_name, std::ios::binary);
	fout.write(bytes, bytes_size);
	delete[] bytes;
}

bool vann::loadFromFile(char *file_name) {
	std::ifstream fin(file_name, std::ios::binary);

	unsigned int temp;

	//number of bytes
	fin.read((char*)&temp, 4);

	//number of layers
	fin.read((char*)&temp, 4);

	unsigned int *totalLayers = new unsigned int[temp];
	fin.read((char*)totalLayers, temp * 4);

	setLayersWithoutInitialising(temp, totalLayers);
	delete[] totalLayers;

	fin.read((char*)&temp, 4);
	if (temp != num_neurons)
		return false;
	fin.read((char*)biases, temp * 8);

	fin.read((char*)&temp, 4);
	if (temp != num_connections)
		return false;
	fin.read((char*)weights, temp * 8);
	
	char checked;
	fin.read(&checked, 1);
	return checked;
}

void vann::destroy() {
	if (usesMemory) {
		max_layer_size = -1;

		delete[] layers;
		delete[] output;
		delete[] derivative;
		delete[] weights;
		delete[] weights_batch_derivative;
		delete[] weights_delta;
		delete[] biases;
		delete[] biases_batch_derivative;
		delete[] aux;
		usesMemory = false;
	}
}

void vann::setLayers(const unsigned int num_layers, ...) {
	va_list argc;

	va_start(argc, num_layers);
	unsigned int *user_layers = new unsigned int[num_layers];

	for (unsigned int i = 0; i < num_layers; i++)
		user_layers[i] = va_arg(argc, int);

	setLayers(num_layers, user_layers);
}

void vann::setLayersWithoutInitialising(const unsigned int num_layers, unsigned int *user_layers) {
	this->num_layers = num_layers;
	layers = new unsigned int[num_layers];

	num_neurons = 0;
	num_connections = 0;
	max_layer_size = 0;
	for (unsigned int i = 0; i < num_layers; i++) {
		layers[i] = user_layers[i];
		num_neurons += layers[i];
		if (i > 0)
			num_connections += layers[i - 1] * layers[i];
		if (layers[i] > max_layer_size)
			max_layer_size = layers[i];
	}


	output = new vann_type[num_neurons];
	biases = new vann_type[num_neurons];
	biases_batch_derivative = new vann_type[num_neurons];

	derivative = new vann_type*[2];
	derivative[0] = new vann_type[max_layer_size];
	derivative[1] = new vann_type[max_layer_size];
	aux = new vann_type[max_layer_size];

	weights = new vann_type[num_connections];
	weights_batch_derivative = new vann_type[num_connections];
	weights_delta = new vann_type[num_connections];
	memset(weights_delta, 0, num_connections * sizeof(vann_type));

	long long seed = time(NULL);
	if (use_custom_seed)
		seed = custom_seed;
	std::cout << "Seed: " << seed << '\n';
	srand(seed);

	usesMemory = true;
}

void vann::setLayers(const unsigned int num_layers, unsigned int *user_layers) {
	setLayersWithoutInitialising(num_layers, user_layers);

	for (int i = 0; i < num_neurons; i++)
		biases[i] = randomBias();
	for (int i = 0; i < num_connections; i++)
		weights[i] = randomWeight();
}

void vann::resetBatchDerivatives() {
	memset(weights_batch_derivative, 0, num_connections*8);
	memset(biases_batch_derivative, 0, num_neurons*8);
}