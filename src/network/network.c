#include "data/data.h"
#include "network/network.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

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
        nn->layers[i]->prev_layer = (i == 0) ? NULL : nn->layers[i - 1];
        nn->layers[i]->next_layer = (i ==  layer_length - 1) ? NULL : nn->layers[i + 1];
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
            matrix_multiply(curr->weights, inputs, curr->weighted_input, nn->inputs_length, curr->length, 1);
        }
        else {
            matrix_multiply(curr->weights, curr->prev_layer->output, curr->weighted_input, curr->prev_layer->length, curr->length, 1);
        }
        for (size_t i = 0; i < curr->length; i++) {
            curr->weighted_input[i] = curr->weighted_input[i] + curr->bias[i];
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
    float *percentages = malloc(output_layer->length * sizeof(float));
    int *indices = malloc(output_layer->length * sizeof(int));

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
        int temp_idx = indices[max_idx];
        indices[max_idx] = indices[i];
        indices[i] = temp_idx;
    }

    // Print the sorted percentages with neuron indices
    for (unsigned int i = 0; i < output_layer->length; i++) {
        printf(" (%d = %.2f%%) ", indices[i], percentages[i]);
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

void layer_learn_output(neural_network_t *nn, layer_t *previous_layer, layer_t *layer, float learn_rate, data_t *data, float (*activation_function)(float, int)) {
    add_inputs(nn, data->inputs);
    compute_network(nn);
    for (unsigned int i = 0; i < layer->length; i++) {
        float neuron_output = layer->neurons[i].output;
        float target_output = output_neuron_expected(i, data);

        layer->neurons[i].delta = 2 * (neuron_output - target_output) * activation_function(layer->neurons[i].weighted_input, 1);

        for (int j = 0; j < layer->neurons[i].num_weights; j++) {
            float input = previous_layer->neurons[j].output;
            layer->neurons[i].weights[j] -= layer->neurons[i].delta * input * learn_rate;
        }

        layer->neurons[i].bias -= layer->neurons[i].delta * learn_rate;
    }
}

void layer_learn_intermediate(layer_t *previous_layer, layer_t *layer, layer_t *next_layer, float learn_rate, float (*activation_function)(float, int)) {
    for (unsigned int i = 0; i < layer->length; i++) {
        layer->neurons[i].delta = 0.0f;
        for (unsigned int j = 0; j < next_layer->length; j++) {
            float weight_next_neuron = next_layer->neurons[j].weights[i];
            float delta_next_neuron = next_layer->neurons[j].delta;
            layer->neurons[i].delta += weight_next_neuron * delta_next_neuron * activation_function(layer->neurons[i].weighted_input, 1);
        }

        for (int j = 0; j < layer->neurons[i].num_weights; j++) {
            float input = previous_layer->neurons[j].output;
            layer->neurons[i].weights[j] -= layer->neurons[i].delta * input * learn_rate;
        }

        layer->neurons[i].bias -= layer->neurons[i].delta * learn_rate;
    }
}

void learn(neural_network_t *nn, float learn_rate, data_t *data) {
    layer_learn_output(nn, (nn->num_hidden_layer == 0) ? &nn->input_layer : &nn->hidden_layers[nn->num_hidden_layer - 1], &nn->output_layer, learn_rate, data, nn->activation_function);
    for (int j = nn->num_hidden_layer - 1; j >= 0; j--) {
        layer_learn_intermediate((j == 0) ? &nn->input_layer : &nn->hidden_layers[j - 1], &nn->hidden_layers[j], ((unsigned int)j == nn->num_hidden_layer - 1) ? &nn->output_layer : &nn->hidden_layers[j + 1], learn_rate, nn->activation_function);
    }
}

float random_float(float min, float max) { return (float)rand() / (float)RAND_MAX * (max - min) + min; }

void save_network(const char *filename, neural_network_t *network) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing\n");
        return;
    }

    fwrite(&(network->input_layer.length), sizeof(unsigned int), 1, file);
    fwrite(&(network->num_hidden_layer), sizeof(unsigned int), 1, file);

    for (unsigned int i = 0; i < network->num_hidden_layer; i++) {
        fwrite(&(network->hidden_layers[i].length), sizeof(int), 1, file);
        for (unsigned int j = 0; j < network->hidden_layers[i].length; j++) {
            fwrite(network->hidden_layers[i].neurons[j].weights, sizeof(float), network->hidden_layers[i].neurons[j].num_weights, file);
            fwrite(&(network->hidden_layers[i].neurons[j].bias), sizeof(float), 1, file);
        }
    }

    // Output layer
    fwrite(&(network->output_layer.length), sizeof(unsigned int), 1, file);
    for (unsigned int i = 0; i < network->output_layer.length; i++) {
        fwrite(network->output_layer.neurons[i].weights, sizeof(float), network->output_layer.neurons[i].num_weights, file);
        fwrite(&(network->output_layer.neurons[i].bias), sizeof(float), 1, file);
    }

    fclose(file);
}

void load_network(const char *filename, neural_network_t *network) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading\n");
        return;
    }

    unsigned int check_val = 0;
    fread(&check_val, sizeof(unsigned int), 1, file);
    if (check_val != network->input_layer.length) {
        printf("Number of input layer not compatiable with save file, read: %d\n", check_val);
        return;
    }
    fread(&check_val, sizeof(unsigned int), 1, file);

    if (check_val != network->num_hidden_layer) {
        printf("Number of hidden layer not compatable with save file, read: %d\n", check_val);
        return;
    }
    for (unsigned int i = 0; i < network->num_hidden_layer; i++) {
        check_val = 0;
        fread(&check_val, sizeof(unsigned int), 1, file);
        if (check_val != network->hidden_layers[i].length) {
            printf("Number of hidden layer neuron not compatable with save file, read: %d\n", check_val);
            return;
        }
        for (unsigned int j = 0; j < network->hidden_layers[i].length; j++) {
            fread(network->hidden_layers[i].neurons[j].weights, sizeof(float), network->hidden_layers[i].neurons[j].num_weights, file);
            fread(&(network->hidden_layers[i].neurons[j].bias), sizeof(float), 1, file);
        }
    }

    // Output layer
    check_val = 0;
    fread(&check_val, sizeof(unsigned int), 1, file);
    if (check_val != network->output_layer.length) {
        printf("Number of output layer neuron not compatable with save file, read: %d\n", check_val);
        return;
    }
    for (unsigned int i = 0; i < network->output_layer.length; i++) {
        fread(network->output_layer.neurons[i].weights, sizeof(float), network->output_layer.neurons[i].num_weights, file);
        fread(&(network->output_layer.neurons[i].bias), sizeof(float), 1, file);
    }
    fclose(file);
}

float test_network_percent(neural_network_t *nn, dataset_t* test_dataset) {
    int correct = 0;
    for (unsigned int i = 0; i < test_dataset->length; i++) {
        add_inputs(nn, test_dataset->datas[i]->inputs);
        compute_network(nn);
        unsigned int max = 0;
        for (int i = 1; i < 10; i++) {
            if (output_neuron_percent_activate(nn, i) > output_neuron_percent_activate(nn, max)) {
                max = i;
            }
        }
        if (max == test_dataset->datas[i]->neuron_index) {
            correct++;
        }
    }

    return (float)correct * 100.0f / (float)test_dataset->length;
}

void train(neural_network_t *network, dataset_t *dataset, dataset_t *test_dataset, float learn_rate, int learn_amount, int log_amount) {
    clock_t start_time = clock();
    for (int i = 0; i < learn_amount; i++) {
        if (i % log_amount == 0 && i != 0) {
            float new_cost = cost(network, test_dataset, 100);
            clock_t elapsed_ms = clock() - start_time;
            float elapsed_s = (float)elapsed_ms / CLOCKS_PER_SEC;
            float speed = (float)log_amount / elapsed_s;
            printf("Learned: %d, cost: %f, elapsed time: %.2fs, speed: %.2f Data/s\n", i, new_cost, elapsed_s, speed);
            start_time = clock();
        }
        data_t *data =  get_data_copy(dataset->datas[rand() % dataset->length], IMAGE_SIZE * IMAGE_SIZE);
        rotate_data(data, IMAGE_SIZE, IMAGE_SIZE, random_float(-5.0f, 5.0f));
        scale_data(data, IMAGE_SIZE, IMAGE_SIZE, random_float(0.9f, 1.1f));
        offset_data(data, IMAGE_SIZE, IMAGE_SIZE, random_float(-3.0f, 3.0f), random_float(-3.0f, 3.0f));
        noise_data(data, IMAGE_SIZE * IMAGE_SIZE, 0.3f, 0.08f);
        learn(network, learn_rate, data);
        free_data(data);
    }
}

