#include <assert.h>
#include <cblas.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_THREADING
#include <pthread.h>
#endif

#include "cneuron/cneuron.h"

neural_network *alloc_neural_network(size_t network_length, const size_t *layers_length, size_t inputs_length) {
    size_t total_float = 0;
    for (size_t i = 0; i < network_length; i++) {
        size_t prev_length = (i == 0) ? inputs_length : layers_length[i - 1];
        total_float += layers_length[i] * 4 + layers_length[i] * prev_length;
    }
    neural_network *nn = calloc(1, sizeof(neural_network) + sizeof(size_t) * (network_length * 3 + 2) + sizeof(float) * total_float);
    if (!nn) return NULL;
    nn->length = network_length;
    nn->inputs_length = inputs_length;
    nn->layer_lengths = (size_t *)(nn + 1);
    nn->prev_lengths_sums = nn->layer_lengths + network_length;
    nn->prev_weights_sums = nn->prev_lengths_sums + network_length + 1;
    size_t lengths_sums = 0;
    size_t weights_sums = 0;
    for (size_t i = 0; i < network_length; i++) {
        nn->prev_lengths_sums[i] = lengths_sums;
        nn->prev_weights_sums[i] = weights_sums;
        lengths_sums += layers_length[i];
        size_t prev_length = (i == 0) ? inputs_length : layers_length[i - 1];
        weights_sums += layers_length[i] * prev_length;
        nn->layer_lengths[i] = layers_length[i];
    }
    nn->prev_lengths_sums[network_length] = lengths_sums;
    nn->prev_weights_sums[network_length] = weights_sums;

    nn->delta = (float *)(nn->prev_weights_sums + network_length + 1);
    nn->weighted_input = nn->delta + lengths_sums;
    nn->output = nn->weighted_input + lengths_sums;
    nn->bias = nn->output + lengths_sums;
    nn->weights = nn->bias + lengths_sums;

    return nn;
}

neural_network *get_neural_network(size_t network_length, const size_t *layers_length, size_t inputs_length, float (*activation_function)(float, bool)) {
    assert(layers_length);

    neural_network *nn = alloc_neural_network(network_length, layers_length, inputs_length);
    if (!nn) return NULL;

    for (size_t i = 0; i < nn->prev_weights_sums[nn->length]; i++) {
        // Initialise weights to -1.0f - 1.0f
        nn->weights[i] = randf(2.0f, -1.0f);
    }

    nn->activation_function = activation_function;
    return nn;
}

void compute_network(const neural_network *restrict nn, const float *restrict inputs) {
    assert(nn && inputs);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, nn->layer_lengths[0], 1, nn->inputs_length, 1.0f, nn->weights, nn->layer_lengths[0], inputs, nn->inputs_length, 0.0f, nn->weighted_input, nn->layer_lengths[0]);
    cblas_saxpy(nn->layer_lengths[0], 1.0f, nn->bias, 1, nn->weighted_input, 1);
    vector_apply_activation(nn->weighted_input, nn->output, nn->layer_lengths[0], nn->activation_function, false);
    for (size_t i = 1; i < nn->length; i++) {
        size_t len = nn->layer_lengths[i];
        size_t prev_len = nn->layer_lengths[i - 1];
        size_t w_sum = nn->prev_weights_sums[i];
        size_t l_sum = nn->prev_lengths_sums[i];
        size_t prev_l_sum = nn->prev_lengths_sums[i - 1];

        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, len, 1, prev_len, 1.0f, &nn->weights[w_sum], len, &nn->output[prev_l_sum], prev_len, 0.0f, &nn->weighted_input[l_sum], len);
        cblas_saxpy(len, 1.0f, &nn->bias[l_sum], 1, &nn->weighted_input[l_sum], 1);
        vector_apply_activation(&nn->weighted_input[l_sum], &nn->output[l_sum], len, nn->activation_function, false);
    }
}

float softmax(const neural_network *nn, size_t neuron_index) {
    assert(nn);

    float sum = 0.0f;
    float max_output = -FLT_MAX;

    // Last layer
    size_t len = nn->layer_lengths[nn->length - 1];
    size_t l_sum = nn->prev_lengths_sums[nn->length - 1];
    for (size_t i = 0; i < len; i++) {
        if (nn->output[l_sum + i] > max_output)
            max_output = nn->output[l_sum + i];
    }

    for (size_t i = 0; i < len; i++)
        sum += expf(nn->output[l_sum + i] - max_output);

    return expf(nn->output[l_sum + neuron_index] - max_output) / sum;
}

void print_activation_percentages(const neural_network *nn) {
    assert(nn);

    size_t len = nn->layer_lengths[nn->length - 1];
    float *percentages = malloc(sizeof(float) * len);
    if (!percentages) return;

    size_t *indices = malloc(sizeof(size_t) * len);
    if (!indices) {
        free(percentages);
        return;
    }

    // Store the activation percentages and indices
    for (size_t i = 0; i < len; i++) {
        percentages[i] = softmax(nn, i) * 100.0f;
        indices[i] = i;
    }

    // Selection sort for percentages and corresponding indices
    for (size_t i = 0; i < len - 1; i++) {
        int max_idx = i;
        for (size_t j = i + 1; j < len; j++) {
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
    for (size_t i = 0; i < len; i++)
        printf(" (%zu = %.2f%%) ", indices[i], percentages[i]);

    printf("\n");

    free(percentages);
    free(indices);
}

float cost(const neural_network *nn, const dataset *test_dataset, size_t num_test) {
    assert(nn && test_dataset);

    float cost = 0.0f;

    size_t len = nn->layer_lengths[nn->length - 1];
    size_t l_sum = nn->prev_lengths_sums[nn->length - 1];
    for (size_t i = 0; i < num_test; i++) {
        uint32_t randnum = randnum_u32(test_dataset->length, 0);
        float *test_data = &test_dataset->all_inputs[randnum * test_dataset->inputs_length];
        compute_network(nn, test_data);
        for (size_t j = 0; j < len; j++) {
            float output = nn->output[l_sum + j];
            cost += (output - (j == test_dataset->expected_indices[randnum])) * (output - (j == test_dataset->expected_indices[randnum]));
        }
    }
    return cost / num_test;
}

void print_result(const neural_network *nn) {
    assert(nn);

    size_t len = nn->layer_lengths[nn->length - 1];
    size_t l_sum = nn->prev_lengths_sums[nn->length - 1];
    for (size_t i = 0; i < len; i++)
        printf("%f ", nn->output[l_sum + i]);
}

void layer_learn(const neural_network *nn, size_t layer_index, float learn_rate, const float *data, const size_t data_expected_index) {
    assert(nn && data);

    size_t len = nn->layer_lengths[layer_index];
    size_t l_sum = nn->prev_lengths_sums[layer_index];
    size_t w_sum = nn->prev_weights_sums[layer_index];
    // f'(Z_i) in weighted_input
    vector_apply_activation(&nn->weighted_input[l_sum], &nn->weighted_input[l_sum], len, nn->activation_function, true);
    if (layer_index == nn->length - 1) {
        // Error in output
        nn->output[l_sum + data_expected_index] -= 1.0f;
    } else {
        // W^T_{i+1}δ_{i+1} in output
        size_t next_len = nn->layer_lengths[layer_index + 1];
        size_t next_l_sum = nn->prev_lengths_sums[layer_index + 1];
        size_t next_w_sum = nn->prev_weights_sums[layer_index + 1];
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, next_len, 1, next_len, 1.0f, &nn->weights[next_w_sum], next_len, &nn->delta[next_l_sum], next_len, 0.0f, &nn->output[l_sum], len);
    }

    hadamard_product(&nn->weighted_input[l_sum], &nn->output[l_sum], &nn->delta[l_sum], len);

    float *weight_gradient;
    if (layer_index == 0) {
        weight_gradient = calloc(len * nn->inputs_length, sizeof(float));
        cblas_sger(CblasColMajor, len, nn->inputs_length, 1.0f, &nn->delta[l_sum], 1, data, 1, weight_gradient, len);
        cblas_saxpy(len * nn->inputs_length, -learn_rate, weight_gradient, 1, &nn->weights[w_sum], 1);
    } else {
        size_t prev_len = nn->layer_lengths[layer_index - 1];
        size_t prev_l_sum = nn->prev_lengths_sums[layer_index - 1];
        weight_gradient = calloc(len * prev_len, sizeof(float));
        cblas_sger(CblasColMajor, len, prev_len, 1.0f, &nn->delta[l_sum], 1, &nn->output[prev_l_sum], 1, weight_gradient, len);
        cblas_saxpy(len * prev_len, -learn_rate, weight_gradient, 1, &nn->weights[w_sum], 1);
    }

    // Bias update
    cblas_saxpy(len, -learn_rate, &nn->delta[l_sum], 1, &nn->bias[l_sum], 1);

    free(weight_gradient);
}

void layer_learn_collect_gradient(const neural_network *nn, float *restrict layer_weights_gradients, float *restrict layer_bias_gradients, size_t layer_index, const float *data, size_t data_expected_index) {
    assert(nn && layer_weights_gradients && layer_bias_gradients && data);

    size_t len = nn->layer_lengths[layer_index];
    size_t l_sum = nn->prev_lengths_sums[layer_index];
    // f'(Z_i) in weighted_input
    vector_apply_activation(&nn->weighted_input[l_sum], &nn->weighted_input[l_sum], len, nn->activation_function, true);
    if (layer_index == nn->length - 1) {
        // Error in output
        nn->output[l_sum + data_expected_index] -= 1.0f;
    } else {
        size_t next_len = nn->layer_lengths[layer_index + 1];
        size_t next_l_sum = nn->prev_lengths_sums[layer_index + 1];
        size_t next_w_sum = nn->prev_weights_sums[layer_index + 1];
        // W^T_{i+1}δ_{i+1} in output
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, len, 1, next_len, 1.0f, &nn->weights[next_w_sum], next_len, &nn->delta[next_l_sum], next_len, 0.0f, &nn->output[l_sum], len);
    }

    hadamard_product(&nn->weighted_input[l_sum], &nn->output[l_sum], &nn->delta[l_sum], len);

    if (layer_index == 0) {
        cblas_sger(CblasColMajor, len, nn->inputs_length, 1.0f, &nn->delta[l_sum], 1, data, 1, layer_weights_gradients, len);
    } else {
        size_t prev_len = nn->layer_lengths[layer_index - 1];
        size_t prev_l_sum = nn->prev_lengths_sums[layer_index - 1];
        cblas_sger(CblasColMajor, len, prev_len, 1.0f, &nn->delta[l_sum], 1, &nn->output[prev_l_sum], 1, layer_weights_gradients, len);
    }

    // Bias update
    cblas_saxpy(len, 1.0f, &nn->delta[l_sum], 1, layer_bias_gradients, 1);
}

void stochastic_gd(const neural_network *nn, float learn_rate, const float *data, size_t data_expected_index) {
    assert(nn && data);

    compute_network(nn, data);
    for (size_t i = 0; i < nn->length; i++) {
        layer_learn(nn, nn->length - i - 1, learn_rate, data, data_expected_index);
    }
}

typedef struct {
    neural_network *nn;
    const dataset *data_batch;
    size_t start;
    size_t end;
    float **weights_gradients;
    float **bias_gradients;
    int thread_index;
} ThreadArgs;

void *thread_worker(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;
    neural_network *nn = args->nn;

    float **weights_gradients = args->weights_gradients;
    float **bias_gradients = args->bias_gradients;

    for (size_t i = 0; i < nn->length; i++) {
        size_t weights_size = nn->layer_lengths[i] * ((i == 0) ? nn->inputs_length : nn->layer_lengths[i - 1]);
        weights_gradients[i] = calloc(weights_size, sizeof(float));
        bias_gradients[i] = calloc(nn->layer_lengths[i], sizeof(float));
    }

    for (size_t i = 0; i < args->data_batch->length; i++) {
        float *data = &args->data_batch->all_inputs[i * args->data_batch->inputs_length];
        compute_network(nn, data);

        for (size_t j = 0; j < nn->length; j++) {
            size_t layer_index = nn->length - j - 1;
            layer_learn_collect_gradient(nn, weights_gradients[layer_index], bias_gradients[layer_index], layer_index, data, args->data_batch->expected_indices[i]);
        }
    }

    return NULL;
}

void mini_batch_gd(neural_network *nn, float learn_rate, const dataset *data_batch) {
    assert(nn && data_batch);

    ThreadArgs args;

    float **weights_gradients = malloc(nn->length * sizeof(float *));
    float **bias_gradients = malloc(nn->length * sizeof(float *));

    args = (ThreadArgs){.nn = nn, .data_batch = data_batch, .weights_gradients = weights_gradients, .bias_gradients = bias_gradients};

#ifdef USE_THREADING
    pthread_t thread;
    pthread_create(&thread, NULL, thread_worker, &args);
    pthread_join(thread, NULL);
#else
    thread_worker(&args);
#endif

    for (size_t i = 0; i < nn->length; i++) {
        size_t len = nn->layer_lengths[i];
        size_t l_sum = nn->prev_lengths_sums[i];
        size_t w_sum = nn->prev_weights_sums[i];
        size_t weights_size = nn->prev_weights_sums[i + 1] - nn->prev_weights_sums[i];
        for (size_t j = 0; j < weights_size; j++) {
            nn->weights[w_sum + j] -= weights_gradients[i][j] / data_batch->length * learn_rate;
        }

        for (size_t j = 0; j < len; j++) {
            nn->bias[l_sum + j] -= (bias_gradients[i][j] / data_batch->length) * learn_rate;
        }
    }

    for (size_t i = 0; i < nn->length; i++) {
        free(weights_gradients[i]);
        free(bias_gradients[i]);
    }
    free(weights_gradients);
    free(bias_gradients);
}

bool save_network(const char *restrict filename, const neural_network *restrict nn) {
    assert(filename && nn);

    return false;
    // FILE *file = fopen(filename, "wb");
    // if (!file) {
    //     fprintf(stderr, "Error opening file '%s' for writing neural network: %s\n", filename, strerror(errno));
    //     return false;
    // }
    //
    // if (fwrite(&(nn->inputs_length), sizeof(uint64_t), 1, file) != 1 ||
    //     fwrite(&(nn->length), sizeof(uint64_t), 1, file) != 1) {
    //     fprintf(stderr, "Failed to write network metadata to '%s'\n", filename);
    //     fclose(file);
    //     return false;
    // }
    //
    // for (size_t i = 0; i < nn->length; i++) {
    //     size_t weights_length = nn->layer[i].length * ((i == 0) ? nn->inputs_length : nn->layers[i - 1].length);
    //
    //     if (fwrite(&(nn->layers[i].length), sizeof(uint64_t), 1, file) != 1 || fwrite(nn->layers[i].weights, sizeof(float), weights_length, file) != weights_length || fwrite(nn->layers[i].bias, sizeof(float), nn->layers[i].length, file) != nn->layers[i].length) {
    //         fprintf(stderr, "Failed to write layer %zu data to '%s'\n", i, filename);
    //         fclose(file);
    //         return false;
    //     }
    // }
    //
    // fclose(file);
    // return true;
}

bool load_network(const char *restrict filename, const neural_network *restrict nn) {
    assert(filename && nn);

    return false;
//     FILE *file = fopen(filename, "rb");
//     if (!file) {
//         fprintf(stderr, "Error opening file '%s' for reading neural network: %s\n", filename, strerror(errno));
//         return false;
//     }
//
//     uint64_t inputs_length = 0;
//     if (fread(&inputs_length, sizeof(uint64_t), 1, file) != 1) {
//         fprintf(stderr, "Failed to read inputs_length from %s\n", filename);
//         goto cleanup;
//     }
//     if (inputs_length != nn->inputs_length) {
//         fprintf(stderr, "Invalid input layer length. Expected: %zu. But found: %llu\n", nn->inputs_length, (unsigned long long)inputs_length);
//         goto cleanup;
//     }
//
//     uint64_t network_length = 0;
//     if (fread(&network_length, sizeof(uint64_t), 1, file) != 1) {
//         fprintf(stderr, "Failed to read network_length from %s\n", filename);
//         goto cleanup;
//     }
//     if (network_length != nn->length) {
//         fprintf(stderr, "Invalid network layer. Expected: %zu. But found: %llu\n", nn->length, (unsigned long long)network_length);
//         goto cleanup;
//     }
//
//     for (size_t i = 0; i < nn->length; i++) {
//         uint64_t layer_length = 0;
//         if (fread(&layer_length, sizeof(uint64_t), 1, file) != 1) {
//             fprintf(stderr, "Failed to read layer_length from %s\n", filename);
//             goto cleanup;
//         }
//         if (layer_length != nn->layers[i].length) {
//             fprintf(stderr, "Invalid layer length. Expected: %zu. But found: %llu\n", nn->layers[i].length, (unsigned long long)layer_length);
//             goto cleanup;
//         }
//
//         size_t weights_length = nn->layers[i].length * ((i == 0) ? nn->inputs_length : nn->layers[i - 1].length);
//         if (fread(nn->layers[i].weights, sizeof(float), weights_length, file) != weights_length || fread(nn->layers[i].bias, sizeof(float), nn->layers[i].length, file) != nn->layers[i].length) {
//             fprintf(stderr, "Failed to read layer %zu data from '%s'\n", i, filename);
//             goto cleanup;
//         }
//     }
//
//     fclose(file);
//     return true;
//
// cleanup:
//     fclose(file);
//     return false;
}

float test_network_percent(const neural_network *nn, const dataset *test_dataset) {
    assert(nn);
    assert(test_dataset);

    int correct = 0;
    for (size_t i = 0; i < test_dataset->length; i++) {
        compute_network(nn, &test_dataset->all_inputs[i * test_dataset->inputs_length]);
        size_t max = 0;
        for (size_t j = 0; j < nn->layer_lengths[nn->length - 1]; j++) {
            if (softmax(nn, j) > softmax(nn, max))
                max = j;
        }
        if (max == test_dataset->expected_indices[i]) {
            correct++;
        }
    }

    return (float)correct * 100.0f / (float)test_dataset->length;
}
