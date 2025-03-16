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
void matrix_multiply(const float *a, const float *b, float *c, size_t rows_a, size_t cols_a, size_t cols_b);

layer_t *get_layer(size_t length, size_t prev_length); 
neural_network_t *get_neural_network(size_t layer_length, const size_t *layer_lengths, size_t inputs_length, float (*activation_function)(float, int));

void free_layer(layer_t *layer);
void free_neural_network(neural_network_t *nn); 

void compute_network(neural_network_t *nn, const float *inputs);

float activation_percentage(neural_network_t *nn, size_t neuron_index);

void print_activation_percentages(neural_network_t *nn);

float output_expected(size_t neuron_index, const data_t *data);

float cost(neural_network_t *nn, const dataset_t *test_dataset, size_t num_test);

void layer_learn_output(layer_t *layer, float learn_rate, const data_t *data, float (*activation_function)(float, int));
void layer_learn_intermediate(layer_t *layer, float learn_rate, const data_t *data, size_t inputs_length, float (*activation_function)(float, int));
void learn(neural_network_t *nn, float learn_rate, const data_t *data);

void save_network(const char *filename, neural_network_t *nn);
void load_network(const char *filename, neural_network_t *nn);

float test_network_percent(neural_network_t *nn, const dataset_t* test_dataset);
