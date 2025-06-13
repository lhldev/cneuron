#include <assert.h>
#include <cblas.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cneuron/cneuron.h"
#include "rand.h"

layer *get_layer(size_t length, size_t prev_length) {
    layer *new_layer = calloc(1, sizeof(layer));
    if (!new_layer)
        return NULL;

    new_layer->length = length;

    new_layer->weights = malloc(sizeof(float) * length * prev_length);
    if (!new_layer->weights) {
        free_layer(new_layer);
        return NULL;
    }

    for (size_t i = 0; i < length * prev_length; i++)
        new_layer->weights[i] = randf(2.0f, -1.0f);

    new_layer->delta = calloc(length, sizeof(float));
    if (!new_layer->delta) {
        free_layer(new_layer);
        return NULL;
    }

    new_layer->bias = calloc(length, sizeof(float));
    if (!new_layer->bias) {
        free_layer(new_layer);
        return NULL;
    }

    new_layer->output = calloc(length, sizeof(float));
    if (!new_layer->output) {
        free_layer(new_layer);
        return NULL;
    }

    new_layer->weighted_input = calloc(length, sizeof(float));
    if (!new_layer->output) {
        free_layer(new_layer);
        return NULL;
    }

    return new_layer;
}

neural_network *get_neural_network(size_t layer_length, const size_t *layer_lengths, size_t inputs_length, float (*activation_function)(float, bool)) {
    assert(layer_lengths);

    neural_network *nn = malloc(sizeof(neural_network));
    if (!nn)
        return NULL;

    // Use calloc for freeing when error
    nn->layers = calloc(layer_length, sizeof(layer));
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

void free_layer(layer *layer) {
    if (!layer)
        return;

    free(layer->weighted_input);
    free(layer->output);
    free(layer->bias);
    free(layer->delta);
    free(layer->weights);
    free(layer);
}

void free_neural_network(neural_network *nn) {
    if (!nn)
        return;

    for (size_t i = 0; i < nn->length; i++)
        free_layer(nn->layers[i]);

    free(nn->layers);
    free(nn);
}

void compute_network(neural_network *nn, const float *inputs) {
    assert(nn && inputs);

    layer *curr = nn->layers[0];
    while (curr != NULL) {
        layer *prev = curr->prev_layer;
        if (prev == NULL) {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, curr->length, 1, nn->inputs_length, 1.0f, curr->weights, curr->length, inputs, nn->inputs_length, 0.0f, curr->weighted_input, curr->length);
        } else {
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, curr->length, 1, prev->length, 1.0f, curr->weights, curr->length, prev->output, prev->length, 0.0f, curr->weighted_input, curr->length);
        }

        cblas_saxpy(curr->length, 1.0f, curr->bias, 1, curr->weighted_input, 1);
        vector_apply_activation(curr->weighted_input, curr->output, curr->length, nn->activation_function, false);
        curr = curr->next_layer;
    }
}

float softmax(neural_network *nn, size_t neuron_index) {
    assert(nn);

    float sum = 0.0f;
    float max_output = -INFINITY;

    layer *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < output_layer->length; i++) {
        if (output_layer->output[i] > max_output)
            max_output = output_layer->output[i];
    }

    for (size_t i = 0; i < output_layer->length; i++)
        sum += expf(output_layer->output[i] - max_output);

    return expf(output_layer->output[neuron_index] - max_output) / sum;
}

void print_activation_percentages(neural_network *nn) {
    assert(nn);

    layer *output_layer = nn->layers[nn->length - 1];
    float *percentages = malloc(sizeof(float) * output_layer->length);
    if (!percentages)
        return;

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
            if (percentages[j] > percentages[max_idx])
                max_idx = j;
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
    for (size_t i = 0; i < output_layer->length; i++)
        printf(" (%zu = %.2f%%) ", indices[i], percentages[i]);

    printf("\n");

    free(percentages);
    free(indices);
}

float cost(neural_network *nn, const dataset *test_dataset, size_t num_test) {
    assert(nn && test_dataset);

    float cost = 0.0f;

    layer *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < num_test; i++) {
        data *test_data = test_dataset->datas[randnum_u32(test_dataset->length, 0)];
        compute_network(nn, test_data->inputs);
        for (size_t j = 0; j < output_layer->length; j++) {
            float output = output_layer->output[j];
            cost += (output - output_expected(j, test_data)) * (output - output_expected(j, test_data));
        }
    }
    return cost / num_test;
}

void print_result(neural_network *nn) {
    assert(nn);

    layer *output_layer = nn->layers[nn->length - 1];
    for (size_t i = 0; i < output_layer->length; i++)
        printf("%f ", output_layer->output[i]);
}

void layer_learn(neural_network *nn, size_t layer_index, float learn_rate, const data *data) {
    assert(nn && data);

    layer *curr_layer = nn->layers[layer_index];
    layer *prev_layer = curr_layer->prev_layer;
    layer *next_layer = curr_layer->next_layer;

    // f'(Z_i) in weighted_input
    vector_apply_activation(curr_layer->weighted_input, curr_layer->weighted_input, curr_layer->length, nn->activation_function, true);
    if (layer_index == nn->length - 1) {
        // Error in output
        curr_layer->output[data->expected_index] -= 1.0f;
    } else {
        // W^T_{i+1}δ_{i+1} in output
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, curr_layer->length, 1, next_layer->length, 1.0f, next_layer->weights, next_layer->length, next_layer->delta, next_layer->length, 0.0f, curr_layer->output, curr_layer->length);
    }

    hadamard_product(curr_layer->weighted_input, curr_layer->output, curr_layer->delta, curr_layer->length);

    float *weight_gradient;
    if (layer_index == 0) {
        weight_gradient = calloc(curr_layer->length * nn->inputs_length, sizeof(float));
        cblas_sger(CblasColMajor, curr_layer->length, nn->inputs_length, 1.0f, curr_layer->delta, 1, data->inputs, 1, weight_gradient, curr_layer->length);
        cblas_saxpy(curr_layer->length * nn->inputs_length, -learn_rate, weight_gradient, 1, curr_layer->weights, 1);
    } else {
        weight_gradient = calloc(curr_layer->length * prev_layer->length, sizeof(float));
        cblas_sger(CblasColMajor, curr_layer->length, prev_layer->length, 1.0f, curr_layer->delta, 1, prev_layer->output, 1, weight_gradient, curr_layer->length);
        cblas_saxpy(curr_layer->length * prev_layer->length, -learn_rate, weight_gradient, 1, curr_layer->weights, 1);
    }

    // Bias update
    cblas_saxpy(curr_layer->length, -learn_rate, curr_layer->delta, 1, curr_layer->bias, 1);

    free(weight_gradient);
}

void layer_learn_collect_gradient(neural_network *nn, float *layer_weights_gradients, float *layer_bias_gradients, size_t layer_index, const data *data) {
    assert(nn && layer_weights_gradients && layer_bias_gradients && data);

    layer *curr_layer = nn->layers[layer_index];
    layer *prev_layer = curr_layer->prev_layer;
    layer *next_layer = curr_layer->next_layer;

    // f'(Z_i) in weighted_input
    vector_apply_activation(curr_layer->weighted_input, curr_layer->weighted_input, curr_layer->length, nn->activation_function, true);
    if (layer_index == nn->length - 1) {
        // Error in output
        curr_layer->output[data->expected_index] -= 1.0f;
    } else {
        // W^T_{i+1}δ_{i+1} in output
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, curr_layer->length, 1, next_layer->length, 1.0f, next_layer->weights, next_layer->length, next_layer->delta, next_layer->length, 0.0f, curr_layer->output, curr_layer->length);
    }

    hadamard_product(curr_layer->weighted_input, curr_layer->output, curr_layer->delta, curr_layer->length);

    if (layer_index == 0) {
        cblas_sger(CblasColMajor, curr_layer->length, nn->inputs_length, 1.0f, curr_layer->delta, 1, data->inputs, 1, layer_weights_gradients, curr_layer->length);
    } else {
        cblas_sger(CblasColMajor, curr_layer->length, prev_layer->length, 1.0f, curr_layer->delta, 1, prev_layer->output, 1, layer_weights_gradients, curr_layer->length);
    }

    // Bias update
    cblas_saxpy(curr_layer->length, 1.0f, curr_layer->delta, 1, layer_bias_gradients, 1);
}

void stochastic_gd(neural_network *nn, float learn_rate, const data *data) {
    assert(nn && data);

    compute_network(nn, data->inputs);
    for (size_t i = 0; i < nn->length; i++) {
        layer_learn(nn, nn->length - i - 1, learn_rate, data);
    }
}

void mini_batch_gd(neural_network *nn, float learn_rate, const dataset *data_batch) {
    assert(nn && data_batch);

    float **weights_gradients = malloc(sizeof(float *) * nn->length);
    float **bias_gradients = malloc(sizeof(float *) * nn->length);

    for (size_t i = 0; i < nn->length; i++) {
        weights_gradients[i] = calloc(nn->layers[i]->length * ((i == 0) ? nn->inputs_length : nn->layers[i - 1]->length), sizeof(float));
        bias_gradients[i] = calloc(nn->layers[i]->length, sizeof(float));
    }

    for (size_t i = 0; i < data_batch->length; i++) {
        data *data = data_batch->datas[i];
        compute_network(nn, data->inputs);
        for (size_t j = 0; j < nn->length; j++) {
            size_t layer_index = nn->length - j - 1;
            layer_learn_collect_gradient(nn, weights_gradients[layer_index], bias_gradients[layer_index], layer_index, data);
        }
    }

    for (size_t i = 0; i < nn->length; i++) {
        for (size_t j = 0; j < nn->layers[i]->length * ((i == 0) ? nn->inputs_length : nn->layers[i - 1]->length); j++)
            nn->layers[i]->weights[j] -= weights_gradients[i][j] / data_batch->length * learn_rate;
        for (size_t j = 0; j < nn->layers[i]->length; j++)
            nn->layers[i]->bias[j] -= bias_gradients[i][j] / data_batch->length * learn_rate;

        free(weights_gradients[i]);
        free(bias_gradients[i]);
    }

    free(weights_gradients);
    free(bias_gradients);
}

bool save_network(const char *filename, neural_network *nn) {
    assert(filename && nn);

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

bool load_network(const char *filename, neural_network *nn) {
    assert(filename && nn);

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

float test_network_percent(neural_network *nn, const dataset *test_dataset) {
    assert(nn);
    assert(test_dataset);

    int correct = 0;
    for (size_t i = 0; i < test_dataset->length; i++) {
        compute_network(nn, test_dataset->datas[i]->inputs);
        size_t max = 0;
        for (size_t j = 0; j < nn->layers[nn->length - 1]->length; j++) {
            if (softmax(nn, j) > softmax(nn, max))
                max = j;
        }
        if (max == test_dataset->datas[i]->expected_index) {
            correct++;
        }
    }

    return (float)correct * 100.0f / (float)test_dataset->length;
}
