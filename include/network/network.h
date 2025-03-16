#pragma once

typedef struct {
    float *delta;   // for backpropagation
    float *weighted_input;
    float *weights; // Column major matrix
    float *bias;
    float *output;
    unsigned int prev_length;
    unsigned int length;
} layer_t;

typedef struct {
    layer_t input_layer;
    layer_t *hidden_layers;
    layer_t output_layer;
    unsigned int num_hidden_layer;
    float (*activation_function)(float, int);
} neural_network_t;

float random_float(float min, float max);

// Temp matrix multiply column major
void matrix_multiply(const float *a, const float *b, float *c, int rows_a, int cols_a, int cols_b) {
    for (int col = 0; col < cols_b; ++col) {
        for (int row = 0; row < rows_a; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < cols_a; ++k) {
                sum += a[k * rows_a + row] * b[col * cols_a + k]; 
            }
            c[col * rows_a + row] = sum;
        }
    }
}

void calc_output_layer(layer_t *previous_layer, layer_t *current_layer, float (*activation_function)(float, int)); 

void initialise_layer(layer_t *layer, int input_size); 

void initialise_neural_network(neural_network_t *nn, int num_hidden_layer, int *hidden_layer_sizes, int output_layer_size, int num_input, float (*activation_function)(float, int)); 

void free_layer(layer_t *layer);

void free_neural_network(neural_network_t *nn); 

void add_inputs(neural_network_t *nn, float *inputs);

void compute_network(neural_network_t *nn);

float output_neuron_percent_activate(neural_network_t *nn, int neuron_index);

void print_output_neurons_percent_activate(neural_network_t *nn);

float output_neuron_expected(unsigned int neuron_index, data_t *data);

float cost(neural_network_t *nn, dataset_t *test_dataset, unsigned int num_test);


void layer_learn_output(neural_network_t *nn, layer_t *previous_layer, layer_t *layer, float learn_rate, data_t *data, float (*activation_function)(float, int));

void layer_learn_intermediate(layer_t *previous_layer, layer_t *layer, layer_t *next_layer, float learn_rate, float (*activation_function)(float, int));

void learn(neural_network_t *nn, float learn_rate, data_t *data);

void save_network(const char *filename, neural_network_t *network);

void load_network(const char *filename, neural_network_t *network);

float test_network_percent(neural_network_t *nn, dataset_t* test_dataset);

void train(neural_network_t *network, dataset_t *dataset, dataset_t *test_dataset, float learn_rate, int learn_amount, int log_amount);
