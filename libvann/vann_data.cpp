#include "vann.h"
#include <fstream>

unsigned int vann_data::size() {
	return v_input.size();
}

void vann_data::addTest(vann_type *input, vann_type* output) {
	v_input.push_back(input);
	v_output.push_back(output);
}

void vann_data::setTest(unsigned int test, vann_type *input, vann_type* output) {
	if (test >= v_input.size ())
		return;

	v_input[test] = input;
	v_output[test] = output;
}

vann_type* vann_data::getInput(unsigned int test) {
	if (test >= v_input.size())
		return NULL;
	return v_input[test];
}

vann_type* vann_data::getOutput(unsigned int test) {
	if (test >= v_output.size())
		return NULL;
	return v_output[test];
}

void vann_data::clear() {
	v_input.clear();
	v_output.clear();
}

void vann_data::set_data_from_file(char* file_name) {
	FILE *f;
	fopen_s(&f, file_name, "r");

	unsigned int T = 0;
	if (f == NULL) {
		perror("File read error");
		return;
	}
	if (fscanf_s(f, "%ld %ld %ld", &T, &input_size, &output_size) == -1)
		return;
	for (int t = 0; t < T; t++) {
		vann_type *input = new vann_type[input_size], *output = new vann_type[output_size];
		for (int i = 0; i < input_size; i++)
			fscanf_s(f, "%lf", &(input[i]));

		for (int i = 0; i < output_size; i++)
			fscanf_s(f, "%lf", &(output[i]));

		addTest(input, output);
	}
	fclose(f);
}