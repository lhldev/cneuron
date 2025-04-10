#include "network/network.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "data/data.h"

float random_float(float min, float max) {
    assert(min < max);

    return (float)rand() / (float)RAND_MAX * (max - min) + min;
}

void matrix_multiply(const float *a, const float *b, float *c, size_t rows_a, size_t cols_a, size_t cols_b) {
    assert(a);
    assert(b);
    assert(c);

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
    layer_t *layer = calloc(1, sizeof(layer_t));
    if (!layer) {
        return NULL;
    }

    layer->length = length;

    layer->weights = malloc(sizeof(float) * length * prev_length);
    if (!layer->weights) {
        free_layer(layer);
        return NULL;
    }

    for (size_t i = 0; i < length * prev_length; i++) {
        layer->weights[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f);
    }

    layer->delta = malloc(sizeof(float) * length);
    if (!layer->delta) {
        free_layer(layer);
        return NULL;
    }

    layer->bias = malloc(sizeof(float) * length);
    if (!layer->bias) {
        free_layer(layer);
        return NULL;
    }

    layer->output = malloc(sizeof(float) * length);
    if (!layer->output) {
        free_layer(layer);
        return NULL;
    }

    layer->weighted_input = malloc(sizeof(float) * length);
    if (!layer->output) {
        free_layer(layer);
        return NULL;
    }

    for (size_t i = 0; i < length; i++) {
        layer->delta[i] = 0.0f;
        layer->bias[i] = 0.0f;
        layer->output[i] = 0.0f;
        layer->weighted_input[i] = 0.0f;
    }

    return layer;
}

neural_network_t *get_neural_network(size_t layer_length, const size_t *layer_lengths, size_t inputs_length, float (*activation_function)(float, bool)) {
    assert(layer_lengths);

    neural_network_t *nn = malloc(sizeof(neural_network_t));
    if (!nn) {
        return NULL;
    }

    nn->layers = calloc(layer_length, sizeof(layer_t));
    if (!nn->layers) {
        free(nn);
        return NULL;
    }

    nn->length = layer_length;
    nn->inputs_length = inputs_length;

    for (size_t i = 0; i < layer_length; i++) {
        nn->layers[i] = get_layer(layer_lengths[i], (i == 0) ? inputs_length : layer_lengths[i - 1]);
        if (!nn->layers[i]) {
            free_neural_network(nn);
            return NULL;
        }
    }

    for (size_t i = 0; i < layer_length; i++) {
        nn->layers[i]->prev_layer = (i == 0) ? NULL : nn->layers[i - 1];
        nn->layers[i]->next_layer = (i == layer_length - 1) ? NULL : nn->layers[i + 1];
    }

    nn->activation_function = activation_function;
    return nn;
}

void free_layer(layer_t *layer) {
    if (!layer) {
        return;
    }

    free(layer->weighted_input);
    free(layer->output);
    free(layer->bias);
    free(layer->delta);
    free(layer->weights);
    free(layer);
}

void free_neural_network(neural_network_t *nn) {
    if (!nn) {
        return;
    }

    for (size_t i = 0; i < nn->length; i++) {
        free_layer(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}

void compute_network(neural_network_t *nn, const float *inputs) {
    assert(nn);
    assert(inputs);

    layer_t *curr = nn->layers[0];
    while (curr != NULL) {
        if (curr->prev_layer == NULL) {
            matrix_multiply(curr->weights, inputs, curr->weighted_input, curr->length, nn->inputs_length, 1);
        } else {
            matrix_multiply(curr->weights, curr->prev_layer->output, curr->weighted_input, curr->length, curr->prev_layer->length, 1);
        }
        for (size_t i = 0; i < curr->length; i++) {
            curr->weighted_input[i] += curr->bias[i];
            curr->output[i] = nn->activation_function(curr->weighted_input[i], 0);
        }
        curr = curr->next_layer;
    }
}

float softmax(neural_network_t *nn, size_t neuron_index) {
    assert(nn);

    float sum = 0.0f;
    float max_output = -INFINITY;

    layer_t *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < output_layer->length; i++) {
        if (output_layer->output[i] > max_output) {
            max_output = output_layer->output[i];
        }
    }

    for (size_t i = 0; i < output_layer->length; i++) {
        sum += expf(output_layer->output[i] - max_output);
    }

    return expf(output_layer->output[neuron_index] - max_output) / sum;
}

void print_activation_percentages(neural_network_t *nn) {
    assert(nn);

    layer_t *output_layer = nn->layers[nn->length - 1];
    float *percentages = malloc(sizeof(float) * output_layer->length);
    if (!percentages) {
        return;
    }

    size_t *indices = malloc(sizeof(size_t) * output_layer->length);
    if (!indices) {
        free(percentages);
        return;
    }

    // Store the activation percentages and indices
    for (size_t i = 0; i < output_layer->length; i++) {
        percentages[i] = softmax(nn, i) * 100.0f;
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

float cost(neural_network_t *nn, const dataset_t *test_dataset, size_t num_test) {
    assert(nn);
    assert(test_dataset);

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
    assert(nn);

    layer_t *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < output_layer->length; i++) {
        printf("%f ", output_layer->output[i]);
    }
}

void layer_learn(neural_network_t *nn, size_t layer_index, float learn_rate, const data_t *data, float (*activation_function)(float, bool)) {
    assert(nn);
    assert(data);
    assert(activation_function);

    if (layer_index == nn->length - 1) {
        // Output layer learn
        layer_t *output_layer = nn->layers[layer_index];
        for (size_t i = 0; i < output_layer->length; i++) {
            float neuron_output = output_layer->output[i];
            float target_output = output_expected(i, data);

            output_layer->delta[i] = 2 * (neuron_output - target_output) * activation_function(output_layer->weighted_input[i], true);

            // If output_layer is the only layer use data as prev_layer
            if (nn->length == 1) {
                for (size_t j = 0; j < nn->inputs_length; j++) {
                    output_layer->weights[j * output_layer->length + i] -= output_layer->delta[i] * data->inputs[j] * learn_rate;
                }
            } else {
                layer_t *prev_layer = output_layer->prev_layer;
                for (size_t j = 0; j < prev_layer->length; j++) {
                    output_layer->weights[j * output_layer->length + i] -= output_layer->delta[i] * prev_layer->output[j] * learn_rate;
                }
            }

            output_layer->bias[i] -= output_layer->delta[i] * learn_rate;
        }
    } else {
        // Intermediate layer learn
        layer_t *curr_layer = nn->layers[layer_index];
        layer_t *prev_layer = curr_layer->prev_layer;
        layer_t *next_layer = curr_layer->next_layer;
        for (size_t i = 0; i < curr_layer->length; i++) {
            curr_layer->delta[i] = 0.0f;
            for (size_t j = 0; j < next_layer->length; j++) {
                float weight_next_neuron = next_layer->weights[i * next_layer->length + j];
                float delta_next_neuron = next_layer->delta[j];
                curr_layer->delta[i] += weight_next_neuron * delta_next_neuron * activation_function(curr_layer->weighted_input[i], true);
            }

            if (prev_layer != NULL) {
                for (size_t j = 0; j < prev_layer->length; j++) {
                    float input = prev_layer->output[j];
                    curr_layer->weights[j * curr_layer->length + i] -= curr_layer->delta[i] * input * learn_rate;
                }
            } else {
                for (size_t j = 0; j < nn->inputs_length; j++) {
                    float input = data->inputs[j];
                    curr_layer->weights[j * curr_layer->length + i] -= curr_layer->delta[i] * input * learn_rate;
                }
            }

            curr_layer->bias[i] -= curr_layer->delta[i] * learn_rate;
        }
    }
}

void learn(neural_network_t *nn, float learn_rate, const data_t *data) {
    assert(nn);
    assert(data);

    compute_network(nn, data->inputs);
    for (size_t i = 0; i < nn->length; i++) {
        layer_learn(nn, nn->length - i - 1, learn_rate, data, nn->activation_function);
    }
}

bool save_network(const char *filename, neural_network_t *nn) {
    assert(filename);
    assert(nn);

    FILE *file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file '%s' for writing neural network: %s\n", filename, strerror(errno));
        return false;
    }

    if (fwrite(&(nn->inputs_length), sizeof(uint64_t), 1, file) != 1 ||
        fwrite(&(nn->length), sizeof(uint64_t), 1, file) != 1) {
        fprintf(stderr, "Failed to write network metadata to '%s'\n", filename);
        fclose(file);
        return false;
    }

    for (size_t i = 0; i < nn->length; i++) {
        size_t weights_length = nn->layers[i]->length * ((i == 0) ? nn->inputs_length : nn->layers[i]->prev_layer->length);

        if (fwrite(&(nn->layers[i]->length), sizeof(uint64_t), 1, file) != 1 || fwrite(nn->layers[i]->weights, sizeof(float), weights_length, file) != weights_length || fwrite(nn->layers[i]->bias, sizeof(float), nn->layers[i]->length, file) != nn->layers[i]->length) {
            fprintf(stderr, "Failed to write layer %zu data to '%s'\n", i, filename);
            fclose(file);
            return false;
        }
    }

    fclose(file);
    return true;
}

bool load_network(const char *filename, neural_network_t *nn) {
    assert(filename);
    assert(nn);

    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file '%s' for reading neural network: %s\n", filename, strerror(errno));
        return false;
    }

    uint64_t inputs_length = 0;
    if (fread(&inputs_length, sizeof(uint64_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read inputs_length from %s\n", filename);
        goto cleanup;
    }
    if (inputs_length != nn->inputs_length) {
        fprintf(stderr, "Invalid input layer length. Expected: %zu. But found: %llu\n", nn->inputs_length, (unsigned long long)inputs_length);
        goto cleanup;
    }

    uint64_t network_length = 0;
    if (fread(&network_length, sizeof(uint64_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read network_length from %s\n", filename);
        goto cleanup;
    }
    if (network_length != nn->length) {
        fprintf(stderr, "Invalid network layer. Expected: %zu. But found: %llu\n", nn->length, (unsigned long long)network_length);
        goto cleanup;
    }

    for (size_t i = 0; i < nn->length; i++) {
        uint64_t layer_length = 0;
        if (fread(&layer_length, sizeof(uint64_t), 1, file) != 1) {
            fprintf(stderr, "Failed to read layer_length from %s\n", filename);
            goto cleanup;
        }
        if (layer_length != nn->layers[i]->length) {
            fprintf(stderr, "Invalid layer length. Expected: %zu. But found: %llu\n", nn->layers[i]->length, (unsigned long long)layer_length);
            goto cleanup;
        }

        size_t weights_length = nn->layers[i]->length * ((i == 0) ? nn->inputs_length : nn->layers[i]->prev_layer->length);
        if (fread(nn->layers[i]->weights, sizeof(float), weights_length, file) != weights_length || fread(nn->layers[i]->bias, sizeof(float), nn->layers[i]->length, file) != nn->layers[i]->length) {
            fprintf(stderr, "Failed to read layer %zu data from '%s'\n", i, filename);
            goto cleanup;
        }
    }

    fclose(file);
    return true;

cleanup:
    fclose(file);
    return false;
}

float test_network_percent(neural_network_t *nn, const dataset_t *test_dataset) {
    assert(nn);
    assert(test_dataset);

    int correct = 0;
    for (size_t i = 0; i < test_dataset->length; i++) {
        compute_network(nn, test_dataset->datas[i]->inputs);
        size_t max = 0;
        for (size_t j = 0; j < nn->layers[nn->length - 1]->length; j++) {
            if (softmax(nn, j) > softmax(nn, max)) {
                max = j;
            }
        }
        if (max == test_dataset->datas[i]->expected_index) {
            correct++;
        }
    }

    return (float)correct * 100.0f / (float)test_dataset->length;
}
