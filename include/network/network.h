#pragma once

#include "data/data.h"

#include <stddef.h>

typedef struct layer layer_t;

typedef struct layer {
    float *delta;   // for backpropagation
    float *weighted_input;
    float *weights; // Column major matrix
    float *bias;
    float *output;
    layer_t *prev_layer;
    layer_t *next_layer;
    size_t length;
} layer_t;

typedef struct {
    layer_t **layers;
    size_t length;
    size_t inputs_length;
    float (*activation_function)(float, int);
} neural_network_t;

float random_float(float min, float max);

// Temp matrix multiply column major
void matrix_multiply(const float *a, const float *b, float *c, size_t rows_a, size_t cols_a, size_t cols_b) {
    for (size_t col = 0; col < cols_b; ++col) {
        for (size_t row = 0; row < rows_a; ++row) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; ++k) {
                sum += a[k * rows_a + row] * b[col * cols_a + k]; 
            }
            c[col * rows_a + row] = sum;
        }
    }
}

layer_t *get_layer(size_t length, size_t prev_length); 
neural_network_t *get_neural_network(size_t layer_length, const size_t *layer_lengths, size_t inputs_length, float (*activation_function)(float, int));

void free_layer(layer_t *layer);
void free_neural_network(neural_network_t *nn); 

void compute_network(neural_network_t *nn, const float *inputs);

float activation_percentage(neural_network_t *nn, size_t neuron_index);

void print_activation_percentages(neural_network_t *nn);

float output_expected(size_t neuron_index, const data_t *data);

float cost(neural_network_t *nn, const dataset_t *test_dataset, size_t num_test);

void layer_learn_output(neural_network_t *nn, layer_t *previous_layer, layer_t *layer, float learn_rate, data_t *data, float (*activation_function)(float, int));
void layer_learn_intermediate(layer_t *previous_layer, layer_t *layer, layer_t *next_layer, float learn_rate, float (*activation_function)(float, int));
void learn(neural_network_t *nn, float learn_rate, data_t *data);

void save_network(const char *filename, neural_network_t *network);
void load_network(const char *filename, neural_network_t *network);

float test_network_percent(neural_network_t *nn, dataset_t* test_dataset);

void train(neural_network_t *network, dataset_t *dataset, dataset_t *test_dataset, float learn_rate, int learn_amount, int log_amount);
