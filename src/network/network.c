#include "data/data.h"
#include "network/network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float random_float(float min, float max) { return (float)rand() / (float)RAND_MAX * (max - min) + min; }

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

layer_t *get_layer(size_t length, size_t prev_length) {
    layer_t *layer = malloc(sizeof(layer_t));

    layer->length = length;

    layer->weights = malloc(sizeof(float) * length * prev_length);
    for (size_t i = 0; i < length * prev_length; i++) {
        layer->weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
    }

    layer->delta = malloc(sizeof(float) * length);
    layer->bias = malloc(sizeof(float) * length);
    layer->output = malloc(sizeof(float) * length);
    layer->weighted_input = malloc(sizeof(float) * length);

    for (size_t i = 0; i < length; i++) {
        layer->delta[i] = 0.0f; 
        layer->bias[i] = 0.0f; 
        layer->output[i] = 0.0f; 
        layer->weighted_input[i] = 0.0f; 
    }

    return layer;
}

neural_network_t *get_neural_network(size_t layer_length, const size_t *layer_lengths, size_t inputs_length, float (*activation_function)(float, int)) {
    neural_network_t *nn = malloc(sizeof(neural_network_t));
    nn->layers = malloc(sizeof(layer_t) * layer_length);
    nn->length = layer_length;
    nn->inputs_length = inputs_length;

    for (size_t i = 0; i < layer_length; i++) {
        nn->layers[i] = get_layer(layer_lengths[i], (i == 0) ? inputs_length : layer_lengths[i - 1]);
    }

    for (size_t i = 0; i < layer_length; i++) {
        nn->layers[i]->prev_layer = (i == 0) ? NULL : nn->layers[i - 1];
        nn->layers[i]->next_layer = (i == layer_length - 1) ? NULL : nn->layers[i + 1];
    }

    nn->activation_function = activation_function;
    return nn;
}

void free_layer(layer_t *layer) {
    free(layer->weighted_input);
    free(layer->output);
    free(layer->bias);
    free(layer->delta);
    free(layer->weights);
    free(layer);
}

void free_neural_network(neural_network_t *nn) {
    for (size_t i = 0; i < nn->length; i++) {
        free_layer(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}

void compute_network(neural_network_t *nn, const float *inputs) {
    layer_t *curr = nn->layers[0];
    while (curr != NULL) {
        if (curr->prev_layer == NULL) {
            matrix_multiply(curr->weights, inputs, curr->weighted_input, curr->length, nn->inputs_length, 1);
        }
        else {
            matrix_multiply(curr->weights, curr->prev_layer->output, curr->weighted_input, curr->length, curr->prev_layer->length, 1);
        }
        for (size_t i = 0; i < curr->length; i++) {
            curr->weighted_input[i] += curr->bias[i];
            curr->output[i] = nn->activation_function(curr->weighted_input[i], 0);
        }
        curr = curr->next_layer;
    }
}

float activation_percentage(neural_network_t *nn, size_t neuron_index) {
    float sum = 0.0f;
    float max_output = -INFINITY;

    layer_t *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < output_layer->length; i++) {
        if (output_layer->output[i] > max_output) {
            max_output = output_layer->output[i];
        }
    }

    for (size_t i = 0; i < output_layer->length; i++) {
        sum += exp(output_layer->output[i] - max_output);
    }

    return exp(output_layer->output[neuron_index] - max_output) / sum * 100.0f;
}

void print_activation_percentages(neural_network_t *nn) {
    layer_t *output_layer = nn->layers[nn->length - 1];
    float *percentages = malloc(sizeof(float) * output_layer->length);
    size_t *indices = malloc(sizeof(size_t) * output_layer->length);

    // Store the activation percentages and indices
    for (size_t i = 0; i < output_layer->length; i++) {
        percentages[i] = activation_percentage(nn, i);
        indices[i] = i;
    }

    // Selection sort for percentages and corresponding indices
    for (size_t i = 0; i < output_layer->length - 1; i++) {
        int max_idx = i;
        for (size_t j = i + 1; j < output_layer->length; j++) {
            if (percentages[j] > percentages[max_idx]) {
                max_idx = j;
            }
        }
        // Swap percentages
        float temp = percentages[max_idx];
        percentages[max_idx] = percentages[i];
        percentages[i] = temp;
        // Swap indices
        size_t temp_idx = indices[max_idx];
        indices[max_idx] = indices[i];
        indices[i] = temp_idx;
    }

    // Print the sorted percentages with neuron indices
    for (size_t i = 0; i < output_layer->length; i++) {
        printf(" (%zu = %.2f%%) ", indices[i], percentages[i]);
    }

    printf("\n");

    free(percentages);
    free(indices);
}

float output_expected(size_t neuron_index, const data_t *data) {
    if (data->neuron_index == neuron_index) {
        return 1.0f;
    } else {
        return 0.0f;
    }
}

float cost(neural_network_t *nn, const dataset_t *test_dataset, size_t num_test) {
    float cost = 0.0f;

    layer_t *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < num_test; i++) {
        data_t *test_data = test_dataset->datas[rand() % test_dataset->length];
        compute_network(nn, test_data->inputs);
        for (size_t j = 0; j < output_layer->length; j++) {
            float output = output_layer->output[j];
            cost += (output - output_expected(j, test_data)) * (output - output_expected(j, test_data));
        }
    }
    return cost / num_test;
}

void print_result(neural_network_t *nn) {
    layer_t *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < output_layer->length; i++) {
        printf("%f ", output_layer->output[i]);
    }
}

void layer_learn_output(layer_t *output_layer, float learn_rate, const data_t *data, float (*activation_function)(float, int)) {
    layer_t *prev_layer = output_layer->prev_layer;
    for (size_t i = 0; i < output_layer->length; i++) {
        float neuron_output = output_layer->output[i];
        float target_output = output_expected(i, data);

        output_layer->delta[i] = 2 * (neuron_output - target_output) * activation_function(output_layer->weighted_input[i], 1);

        for (size_t j = 0; j < prev_layer->length; j++) {
            output_layer->weights[j * output_layer->length + i] -= output_layer->delta[i] * prev_layer->output[j] * learn_rate;
        }

        output_layer->bias[i] -= output_layer->delta[i] * learn_rate;
    }
}

void layer_learn_intermediate(layer_t *curr_layer, float learn_rate, const data_t *data, size_t inputs_length, float (*activation_function)(float, int)) {
    layer_t *prev_layer = curr_layer->prev_layer;
    layer_t *next_layer = curr_layer->next_layer;
    for (size_t i = 0; i < curr_layer->length; i++) {
        curr_layer->delta[i] = 0.0f;
        for (size_t j = 0; j < next_layer->length; j++) {
            float weight_next_neuron = next_layer->weights[i * next_layer->length + j];
            float delta_next_neuron = next_layer->delta[j];
            curr_layer->delta[i] += weight_next_neuron * delta_next_neuron * activation_function(curr_layer->weighted_input[i], 1);
        }

        if (prev_layer != NULL) {
            for (size_t j = 0; j < prev_layer->length; j++) {
                float input = prev_layer->output[j];
                curr_layer->weights[j * curr_layer->length + i] -= curr_layer->delta[i] * input * learn_rate;
            }
        }
        else {
            for (size_t j = 0; j < inputs_length; j++) {
                float input = data->inputs[j];
                curr_layer->weights[j * curr_layer->length + i] -= curr_layer->delta[i] * input * learn_rate;
            }
        }

        curr_layer->bias[i] -= curr_layer->delta[i] * learn_rate;
    }
}

void learn(neural_network_t *nn, float learn_rate, const data_t *data) {
    compute_network(nn, data->inputs);
    for (size_t i = 0; i < nn->length; i++) {
        if (i == 0) {
            layer_learn_output(nn->layers[nn->length - 1], learn_rate, data, nn->activation_function);
        }
        else {
            layer_learn_intermediate(nn->layers[nn->length - i - 1], learn_rate, data, nn->inputs_length, nn->activation_function);
        }
    }
}

void save_network(const char *filename, neural_network_t *nn) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing\n");
        return;
    }

    fwrite(&(nn->inputs_length), sizeof(size_t), 1, file);
    fwrite(&(nn->length), sizeof(size_t), 1, file);

    for (size_t i = 0; i < nn->length; i++) {
        fwrite(&(nn->layers[i]->length), sizeof(size_t), 1, file);
        fwrite(nn->layers[i]->weights, sizeof(float), nn->layers[i]->length * ((i == 0) ? nn->inputs_length : nn->layers[i]->prev_layer->length), file);
        fwrite(nn->layers[i]->bias, sizeof(float), nn->layers[i]->length, file);
    }

    fclose(file);
}

void load_network(const char *filename, neural_network_t *nn) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading\n");
        return;
    }

    size_t check_val = 0;
    fread(&check_val, sizeof(size_t), 1, file);
    if (check_val != nn->inputs_length) {
        printf("Number of input layer not compatiable with save file, read: %zu\n", check_val);
        return;
    }

    fread(&check_val, sizeof(size_t), 1, file);
    if (check_val != nn->length) {
        printf("Number of hidden layer not compatable with save file, read: %zu\n", check_val);
        return;
    }

    for (size_t i = 0; i < nn->length; i++) {
        fread(&check_val, sizeof(size_t), 1, file);
        if (check_val != nn->layers[i]->length) {
            printf("Number of hidden layer neuron not compatable with save file, read: %zu\n", check_val);
            return;
        }

        fread(nn->layers[i]->weights, sizeof(float), nn->layers[i]->length * ((i == 0) ? nn->inputs_length : nn->layers[i]->prev_layer->length), file);
        fread(nn->layers[i]->bias, sizeof(float), nn->layers[i]->length, file);
    }

    fclose(file);
}

float test_network_percent(neural_network_t *nn, const dataset_t* test_dataset) {
    int correct = 0;
    for (size_t i = 0; i < test_dataset->length; i++) {
        compute_network(nn, test_dataset->datas[i]->inputs);
        size_t max = 0;
        for (int i = 1; i < 10; i++) {
            if (activation_percentage(nn, i) > activation_percentage(nn, max)) {
                max = i;
            }
        }
        if (max == test_dataset->datas[i]->neuron_index) {
            correct++;
        }
    }

    return (float)correct * 100.0f / (float)test_dataset->length;
}
