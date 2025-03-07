#include "data/data.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define IMAGE_SIZE 28

typedef struct {
    float delta;  // for backpropagation
    float weighted_input;
    float *weights;
    int num_weights;
    float bias;
    float output;
} neuron_t;

typedef struct {
    unsigned int length;
    neuron_t *neurons;
} layer_t;

typedef struct {
    layer_t input_layer;
    layer_t *hidden_layers;
    layer_t output_layer;
    unsigned int num_hidden_layer;
    float (*activation_function)(float, int);
} neural_network_t;

float sigmoid(float val, int is_deravative) {
    float result = 1 / (1 + exp(-val));
    if (is_deravative == 1) {
        return result * (1 - result);
    }
    return result;
}

float relu(float val, int is_deravative) {
    if (is_deravative) {
        return (val > 0) ? 1 : 0;
    }
    return max(0, val);
}

void calc_output(layer_t *previous_layer, neuron_t *neuron, float (*activation_function)(float, int)) {
    neuron->output = 0.0;
    neuron->weighted_input = 0.0;
    for (unsigned int i = 0; i < previous_layer->length; i++) {
        neuron->weighted_input += previous_layer->neurons[i].output * neuron->weights[i];
    }
    neuron->weighted_input += neuron->bias;
    neuron->output = activation_function(neuron->weighted_input, 0);
}

void calc_output_layer(layer_t *previous_layer, layer_t *current_layer, float (*activation_function)(float, int)) {
    for (unsigned int i = 0; i < current_layer->length; i++) {
        calc_output(previous_layer, &current_layer->neurons[i], activation_function);
    }
}

void initialise_layer(layer_t *layer, int input_size) {
    for (unsigned int i = 0; i < layer->length; i++) {
        layer->neurons[i].weights = malloc(sizeof(float) * input_size);
        layer->neurons[i].num_weights = input_size;
        for (int j = 0; j < input_size; j++) {
            layer->neurons[i].weights[j] = ((float)rand() / RAND_MAX * 2 - 1);
        }
        layer->neurons[i].delta = 0.0;
        layer->neurons[i].bias = 0.0;
        layer->neurons[i].output = 0.0;
        layer->neurons[i].weighted_input = 0.0;
    }
}

void initialise_neural_network(neural_network_t *nn, int num_hidden_layer, int *hidden_layer_sizes, int output_layer_size, int num_input, float (*activation_function)(float, int)) {
    nn->input_layer.neurons = malloc(sizeof(neuron_t) * num_input);
    nn->input_layer.length = num_input;
    nn->hidden_layers = malloc(sizeof(layer_t) * num_hidden_layer);
    nn->num_hidden_layer = num_hidden_layer;

    for (int i = 0; i < num_hidden_layer; i++) {
        nn->hidden_layers[i].neurons = malloc(sizeof(neuron_t) * hidden_layer_sizes[i]);
        nn->hidden_layers[i].length = hidden_layer_sizes[i];
    }

    nn->output_layer.neurons = malloc(sizeof(neuron_t) * output_layer_size);
    nn->output_layer.length = output_layer_size;

    for (int i = 0; i < num_hidden_layer; i++) {
        initialise_layer(&nn->hidden_layers[i], (i == 0) ? num_input : hidden_layer_sizes[i - 1]);
    }
    initialise_layer(&nn->input_layer, num_input);
    initialise_layer(&nn->output_layer, (num_hidden_layer == 0) ? num_input : hidden_layer_sizes[num_hidden_layer - 1]);
    nn->activation_function = activation_function;
}

void free_layer(layer_t *layer) {
    for (unsigned int i = 0; i < layer->length; i++) {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
}

void free_neural_network(neural_network_t *nn) {
    for (unsigned int i = 0; i < nn->num_hidden_layer; i++) {
        free_layer(&nn->hidden_layers[i]);
    }
    free_layer(&nn->output_layer);
    free_layer(&nn->input_layer);
    free(nn->hidden_layers);
}

void add_inputs(neural_network_t *nn, float *inputs) {
    for (unsigned int i = 0; i < nn->input_layer.length; i++) {
        nn->input_layer.neurons[i].output = inputs[i];
    }
}

void compute_network(neural_network_t *nn) {
    if (nn->num_hidden_layer == 0) {
        calc_output_layer(&nn->input_layer, &nn->output_layer, nn->activation_function);
    } else {
        layer_t *curr_layer = &nn->input_layer;
        for (unsigned int i = 0; i < nn->num_hidden_layer; i++) {
            calc_output_layer(curr_layer, &nn->hidden_layers[i], nn->activation_function);
            curr_layer = &nn->hidden_layers[i];
        }
        calc_output_layer(curr_layer, &nn->output_layer, nn->activation_function);
    }
}

float output_neuron_percent_activate(neural_network_t *nn, int neuron_index) {
    float sum = 0.0;
    float max_output = -INFINITY;

    for (unsigned int i = 0; i < nn->output_layer.length; i++) {
        if (nn->output_layer.neurons[i].output > max_output) {
            max_output = nn->output_layer.neurons[i].output;
        }
    }

    for (unsigned int i = 0; i < nn->output_layer.length; i++) {
        sum += exp(nn->output_layer.neurons[i].output - max_output);
    }

    return exp(nn->output_layer.neurons[neuron_index].output - max_output) / sum * 100;
}

void print_output_neurons_percent_activate(neural_network_t *nn) {
    float *percentages = malloc(nn->output_layer.length * sizeof(float));
    int *indices = malloc(nn->output_layer.length * sizeof(int));
    if (percentages == NULL || indices == NULL) {
        printf("Memory allocation failed\n");
        return;
    }

    // Store the activation percentages and indices
    for (unsigned int i = 0; i < nn->output_layer.length; i++) {
        percentages[i] = output_neuron_percent_activate(nn, i);
        indices[i] = i;
    }

    // Selection sort for percentages and corresponding indices
    for (unsigned int i = 0; i < nn->output_layer.length - 1; i++) {
        int max_idx = i;
        for (unsigned int j = i + 1; j < nn->output_layer.length; j++) {
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
    for (unsigned int i = 0; i < nn->output_layer.length; i++) {
        printf(" (%d = %.2f%%) ", indices[i], percentages[i]);
    }

    printf("\n");

    free(percentages);
    free(indices);
}

float output_neuron_expected(unsigned int neuron_index, data_t *data) {
    if (data->neuron_index == neuron_index) {
        return 1.0;
    } else {
        return 0.0;
    }
}

float cost(neural_network_t *nn, data_t *training_data, int num_data) {
    float cost = 0;

    for (int i = 0; i < num_data; i++) {
        add_inputs(nn, training_data[i].inputs);
        compute_network(nn);
        for (unsigned int j = 0; j < nn->output_layer.length; j++) {
            float output = nn->output_layer.neurons[j].output;
            cost += (output - output_neuron_expected(j, &training_data[i])) * (output - output_neuron_expected(j, &training_data[i]));
        }
    }
    return cost / num_data;
}

void print_result(neural_network_t *nn) {
    for (unsigned int i = 0; i < nn->output_layer.length; i++) {
        printf("%f ", nn->output_layer.neurons[i].output);
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
        layer->neurons[i].delta = 0.0;
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

unsigned char *rotate_image(unsigned char *image, int width, int height, float angle) {
    float rad = angle * M_PI / 180.0f;
    float cos_angle = cos(rad);
    float sin_angle = sin(rad);
    unsigned char *new_image = (unsigned char *)malloc(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int center_x = width / 2;
            int center_y = height / 2;
            int src_x = (int)((x - center_x) * cos_angle - (y - center_y) * sin_angle + center_x);
            int src_y = (int)((x - center_x) * sin_angle + (y - center_y) * cos_angle + center_y);

            if (src_y >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                new_image[y * width + x] = image[src_y * width + src_x];
            } else {
                new_image[y * width + x] = 0.0;  // Set background color to black
            }
        }
    }
    return new_image;
}

unsigned char *scale_image(unsigned char *image, int width, int height, float scale) {
    int scale_width = width * scale;
    int scale_height = height * scale;
    unsigned char *scale_image = (unsigned char *)malloc(scale_width * scale_height);
    unsigned char *new_image = (unsigned char *)malloc(width * height);

    for (int y = 0; y < scale_height; y++) {
        for (int x = 0; x < scale_width; x++) {
            int src_x = (int)(x / scale);
            int src_y = (int)(y / scale);

            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                scale_image[y * scale_width + x] = image[src_y * width + src_x];
            } else {
                scale_image[y * scale_width + x] = 0.0;  // Set background color to black
            }
        }
    }
    int off_set_x = (scale_width - width) / 2;
    int off_set_y = (scale_height - height) / 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int scale_x = x + off_set_x;
            int scale_y = y + off_set_y;
            if (scale_x >= 0 && scale_x < scale_width && scale_y >= 0 && scale_y < scale_height) {
                new_image[y * width + x] = scale_image[scale_y * scale_width + scale_x];
            } else {
                new_image[y * width + x] = 0.0;
            }
        }
    }

    free(scale_image);
    return new_image;
}

unsigned char *add_offset(unsigned char *image, int width, int height, int offset_x, int offset_y) {
    unsigned char *new_image = (unsigned char *)malloc(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int new_x = x + offset_x;
            int new_y = y + offset_y;

            if (new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                new_image[y * width + x] = image[new_y * width + new_x];
            } else {
                new_image[y * width + x] = 0.0;  // Set background color to black
            }
        }
    }
    return new_image;
}

unsigned char *add_noise(unsigned char *image, int width, int height, float noise_factor, float probability) {
    unsigned char *new_image = (unsigned char *)malloc(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float random_value = (float)rand() / RAND_MAX;  // Generate a random value between 0 and 1
            if (random_value <= probability) {
                int noise = (int)(rand() % 256 * noise_factor);
                int new_value = image[y * width + x] + noise;

                if (new_value < 0) new_value = 0;
                if (new_value > 255) new_value = 255;

                new_image[y * width + x] = new_value;
            } else {
                new_image[y * width + x] = image[y * width + x];
            }
        }
    }
    return new_image;
}

// data_t data_from_image(char *path, float angle, float scale, int offset_x, int offset_y, float noise_factor, float probability) {
    // data_t data;

    // int width, height, channels;
    // unsigned char *image = stbi_load(path, &width, &height, &channels, STBI_grey);
    // if (image == NULL) {
        // fprintf(stderr, "Failed to load image: %s\n", path);
        // return data;
    // }

    // if (width != IMAGE_SIZE || height != IMAGE_SIZE || channels != 1) {
        // fprintf(stderr, "Invalid image dimensions or channels: %s\n", path);
        // stbi_image_free(image);
        // return data;
    // }

    // unsigned char *scaled_image = scale_image(image, width, height, scale);
    // unsigned char *rotated_image = rotate_image(scaled_image, width, height, angle);
    // unsigned char *offset_image = add_offset(rotated_image, width, height, offset_x, offset_y);
    // unsigned char *noisy_image = add_noise(offset_image, width, height, noise_factor, probability);

    // for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        // data.inputs[i] = (float)noisy_image[i] / 255.0;
    // }

    // char *filename = strrchr(path, '/');
    // if (filename == NULL) {
        // filename = path;
    // } else {
        // filename--;
    // }
    // data.expected = atoi(filename);

    // stbi_image_free(image);
    // free(scaled_image);
    // free(rotated_image);
    // free(offset_image);
    // free(noisy_image);
    // return data;
// }

float random_float(float min, float max) { return (float)rand() / RAND_MAX * (max - min) + min; }

void save_network(const char *filename, neural_network_t *network) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
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
    if (file == NULL) {
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

float test_network_percent(neural_network_t *nn) {
    unsigned int dataset_length = 0;
    unsigned int inputs_length = 0;
    data_t **test_dataset = get_dataset("data/mnist/mnist_test.dat", &dataset_length, &inputs_length);
    int correct = 0;
    for (unsigned int i = 0; i < dataset_length; i++) {
        add_inputs(nn, test_dataset[i]->inputs);
        compute_network(nn);
        unsigned int max = 0;
        for (int i = 1; i < 10; i++) {
            if (output_neuron_percent_activate(nn, i) > output_neuron_percent_activate(nn, max)) {
                max = i;
            }
        }
        if (max == test_dataset[i]->neuron_index) {
            correct++;
        }
    }

    free_dataset(test_dataset, dataset_length);

    return (float)correct * 100.0 / (float)dataset_length;
}

void train(neural_network_t *network, data_t **dataset, unsigned int dataset_length, float learn_rate, int learn_amount, int epoch_amount) {
    clock_t start_time = clock();
    for (int i = 0; i < learn_amount; i++) {
        if (i % epoch_amount == 0 && i != 0) {
            // float new_cost = cost(network, training_data, *num_data);
            clock_t elapsed_ms = clock() - start_time;
            float elapsed_s = (float)elapsed_ms / CLOCKS_PER_SEC;
            float speed = (float)epoch_amount / elapsed_s;
            // printf("Epoch learned %d, cost: %f, elapsed time: %.2fs, speed: %.2f Data/s \n", i, new_cost, elapsed_s, speed);
            printf("Epoch learned %d, elapsed time: %.2fs, speed: %.2f Data/s \n", i, elapsed_s, speed);
            start_time = clock();
        }
        data_t *data =  dataset[rand() % dataset_length];
        learn(network, learn_rate, data);
    }
}

// TODO:
// gpu parallelization

int main() {
    srand(time(NULL));
    unsigned int inputs_length = 0;
    unsigned int dataset_length = 0;
    data_t **dataset = get_dataset("data/mnist/mnist_train.dat", &dataset_length, &inputs_length);
    int num_hidden_layer = 2;
    int *hidden_layer_sizes = malloc(num_hidden_layer * sizeof(int));
    hidden_layer_sizes[0] = 100;
    hidden_layer_sizes[1] = 16;
    int output_layer_size = 10;

    neural_network_t network;
    initialise_neural_network(&network, num_hidden_layer, hidden_layer_sizes, output_layer_size, inputs_length, &sigmoid);

    // Parameters
    float learn_rate = 0.03;
    int learn_amount = 1000000;
    int epoch_amount = 2000;

    char cmd[100];
    FILE *fp;
    float user_input[IMAGE_SIZE * IMAGE_SIZE];
    while (1) {
        printf("cmd: ");
        if (scanf("%99s", cmd) != 1) {
            printf("Invalid input format. Please try again.\n");
            continue;
        }
        if (cmd[0] == 'q') {
            break;
        } else if (cmd[0] == 's') {
            save_network("output/nn.dat", &network);
            printf("Neural network saved!\n");
        } else if (cmd[0] == 'l') {
            load_network("output/nn.dat", &network);
            printf("Neural network loaded!\n");
        } else if (cmd[0] == 't') {
            train(&network, dataset, dataset_length, learn_rate, learn_amount, epoch_amount);
            printf("Training completed. Trained for %d times.\n", learn_amount);
        } else if (cmd[0] == 'T') {
            printf("Testing neural network...\n");
            printf("Network is %.2f%% correct!\n", test_network_percent(&network));
        } else if (cmd[0] == 'i') {
            printf("Enter your input in the window and press enter...\n");
            while (1) {
                system("python3 -W ignore input.py");

                // Open the file for reading
                fp = fopen("output/grid_array.txt", "r");
                if (fp == NULL) {
                    printf("Error opening file\n");
                    break;
                }

                char quit_flag;
                fscanf(fp, " %c", &quit_flag);
                printf("%c\n", quit_flag);
                if (quit_flag == 'q') {
                    fclose(fp);
                    break;
                }

                float generic_float = 0.0;
                int count = 0;
                while (fscanf(fp, "%f", &generic_float) == 1) {
                    if (count >= IMAGE_SIZE * IMAGE_SIZE) {
                        printf("Warning parsing input\n");
                        break;
                    }
                    printf("%.1f ", generic_float);
                    if(count % IMAGE_SIZE == IMAGE_SIZE - 1){
                        printf("\n");
                    }
                    user_input[count++] = generic_float;
                }
                fclose(fp);

                add_inputs(&network, user_input);
                compute_network(&network);
                print_output_neurons_percent_activate(&network);
            }
        } else {
            printf(
                "Command not recognised. \nt - train \nT - test network\ni - insert input \nl - load \ns - save \nq - "
                "quit\n");
            continue;
        }
    }
    free_dataset(dataset, dataset_length);
    free_neural_network(&network);
    if (num_hidden_layer > 0) {
        free(hidden_layer_sizes);
    }
    return 0;
}
