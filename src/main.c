#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define IMAGE_SIZE 28
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    double delta;  // for backpropagation
    double weighted_input;
    double *weights;
    int num_weights;
    double bias;
    double output;
} neuron_t;

typedef struct {
    int size;
    neuron_t *neurons;
} layer_t;

typedef struct {
    layer_t input_layer;
    layer_t *hidden_layers;
    layer_t output_layer;
    int num_hidden_layer;
    double (*activation_function)(double, int);
} neural_network_t;

typedef struct {
    double inputs[IMAGE_SIZE * IMAGE_SIZE];
    int expected;
} data_t;

double sigmoid(double val, int is_deravative) {
    double result = 1 / (1 + exp(-val));
    if (is_deravative == 1) {
        return result * (1 - result);
    }
    return result;
}

double relu(double val, int is_deravative) {
    if (is_deravative) {
        return (val > 0) ? 1 : 0;
    }
    return max(0, val);
}

void calc_output(layer_t *previous_layer, neuron_t *neuron, double (*activation_function)(double, int)) {
    neuron->output = 0.0;
    neuron->weighted_input = 0.0;
    for (int i = 0; i < previous_layer->size; i++) {
        neuron->weighted_input += previous_layer->neurons[i].output * neuron->weights[i];
    }
    neuron->weighted_input += neuron->bias;
    neuron->output = activation_function(neuron->weighted_input, 0);
}

void calc_output_layer(layer_t *previous_layer, layer_t *current_layer, double (*activation_function)(double, int)) {
    for (int i = 0; i < current_layer->size; i++) {
        calc_output(previous_layer, &current_layer->neurons[i], activation_function);
    }
}

void initialise_layer(layer_t *layer, int input_size) {
    for (int i = 0; i < layer->size; i++) {
        layer->neurons[i].weights = malloc(sizeof(double) * input_size);
        layer->neurons[i].num_weights = input_size;
        for (int j = 0; j < input_size; j++) {
            layer->neurons[i].weights[j] = ((double)rand() / RAND_MAX * 2 - 1);
        }
        layer->neurons[i].delta = 0.0;
        layer->neurons[i].bias = 0.0;
        layer->neurons[i].output = 0.0;
        layer->neurons[i].weighted_input = 0.0;
    }
}

void initialise_neural_network(neural_network_t *nn, int num_hidden_layer, int *hidden_layer_sizes, int output_layer_size, int num_input, double (*activation_function)(double, int)) {
    nn->input_layer.neurons = malloc(sizeof(neuron_t) * num_input);
    nn->input_layer.size = num_input;
    nn->hidden_layers = malloc(sizeof(layer_t) * num_hidden_layer);
    nn->num_hidden_layer = num_hidden_layer;

    for (int i = 0; i < num_hidden_layer; i++) {
        nn->hidden_layers[i].neurons = malloc(sizeof(neuron_t) * hidden_layer_sizes[i]);
        nn->hidden_layers[i].size = hidden_layer_sizes[i];
    }

    nn->output_layer.neurons = malloc(sizeof(neuron_t) * output_layer_size);
    nn->output_layer.size = output_layer_size;

    for (int i = 0; i < num_hidden_layer; i++) {
        initialise_layer(&nn->hidden_layers[i], (i == 0) ? num_input : hidden_layer_sizes[i - 1]);
    }
    initialise_layer(&nn->input_layer, num_input);
    initialise_layer(&nn->output_layer, (num_hidden_layer == 0) ? num_input : hidden_layer_sizes[num_hidden_layer - 1]);
    nn->activation_function = activation_function;
}

void free_layer(layer_t *layer) {
    for (int i = 0; i < layer->size; i++) {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
}

void free_neural_network(neural_network_t *nn) {
    for (int i = 0; i < nn->num_hidden_layer; i++) {
        free_layer(&nn->hidden_layers[i]);
    }
    free_layer(&nn->output_layer);
    free_layer(&nn->input_layer);
    free(nn->hidden_layers);
}

void add_inputs(neural_network_t *nn, double *inputs) {
    for (int i = 0; i < nn->input_layer.size; i++) {
        nn->input_layer.neurons[i].output = inputs[i];
    }
}

void compute_network(neural_network_t *nn) {
    if (nn->num_hidden_layer == 0) {
        calc_output_layer(&nn->input_layer, &nn->output_layer, nn->activation_function);
    } else {
        layer_t *curr_layer = &nn->input_layer;
        for (int i = 0; i < nn->num_hidden_layer; i++) {
            calc_output_layer(curr_layer, &nn->hidden_layers[i], nn->activation_function);
            curr_layer = &nn->hidden_layers[i];
        }
        calc_output_layer(curr_layer, &nn->output_layer, nn->activation_function);
    }
}

double output_neuron_percent_activate(neural_network_t *nn, int neuron_index) {
    double sum = 0.0;
    double max_output = -INFINITY;

    for (int i = 0; i < nn->output_layer.size; i++) {
        if (nn->output_layer.neurons[i].output > max_output) {
            max_output = nn->output_layer.neurons[i].output;
        }
    }

    for (int i = 0; i < nn->output_layer.size; i++) {
        sum += exp(nn->output_layer.neurons[i].output - max_output);
    }

    return exp(nn->output_layer.neurons[neuron_index].output - max_output) / sum * 100;
}

void print_output_neurons_percent_activate(neural_network_t *nn) {
    double *percentages = malloc(nn->output_layer.size * sizeof(double));
    int *indices = malloc(nn->output_layer.size * sizeof(int));
    if (percentages == NULL || indices == NULL) {
        printf("Memory allocation failed\n");
        return;
    }

    // Store the activation percentages and indices
    for (int i = 0; i < nn->output_layer.size; i++) {
        percentages[i] = output_neuron_percent_activate(nn, i);
        indices[i] = i;
    }

    // Selection sort for percentages and corresponding indices
    for (int i = 0; i < nn->output_layer.size - 1; i++) {
        int max_idx = i;
        for (int j = i + 1; j < nn->output_layer.size; j++) {
            if (percentages[j] > percentages[max_idx]) {
                max_idx = j;
            }
        }
        // Swap percentages
        double temp = percentages[max_idx];
        percentages[max_idx] = percentages[i];
        percentages[i] = temp;
        // Swap indices
        int temp_idx = indices[max_idx];
        indices[max_idx] = indices[i];
        indices[i] = temp_idx;
    }

    // Print the sorted percentages with neuron indices
    for (int i = 0; i < nn->output_layer.size; i++) {
        printf(" (%d = %.2f%%) ", indices[i], percentages[i]);
    }

    printf("\n");

    free(percentages);
    free(indices);
}

double output_neuron_expected(int neuron_index, data_t *data) {
    if (data->expected == neuron_index) {
        return 1.0;
    } else {
        return 0.0;
    }
}

double cost(neural_network_t *nn, data_t *training_data, int num_data) {
    double cost = 0;

    for (int i = 0; i < num_data; i++) {
        add_inputs(nn, training_data[i].inputs);
        compute_network(nn);
        for (int j = 0; j < nn->output_layer.size; j++) {
            double output = nn->output_layer.neurons[j].output;
            cost += (output - output_neuron_expected(j, &training_data[i])) * (output - output_neuron_expected(j, &training_data[i]));
        }
    }
    return cost / num_data;
}

void print_result(neural_network_t *nn) {
    for (int i = 0; i < nn->output_layer.size; i++) {
        printf("%f ", nn->output_layer.neurons[i].output);
    }
}

void layer_learn_output(neural_network_t *nn, layer_t *previous_layer, layer_t *layer, double learn_rate, data_t *training_data, double (*activation_function)(double, int)) {
    add_inputs(nn, training_data->inputs);
    compute_network(nn);
    for (int i = 0; i < layer->size; i++) {
        double neuron_output = layer->neurons[i].output;
        double target_output = output_neuron_expected(i, training_data);

        layer->neurons[i].delta = 2 * (neuron_output - target_output) * activation_function(layer->neurons[i].weighted_input, 1);

        for (int j = 0; j < layer->neurons[i].num_weights; j++) {
            double input = previous_layer->neurons[j].output;
            layer->neurons[i].weights[j] -= layer->neurons[i].delta * input * learn_rate;
        }

        layer->neurons[i].bias -= layer->neurons[i].delta * learn_rate;
    }
}

void layer_learn_intermediate(layer_t *previous_layer, layer_t *layer, layer_t *next_layer, double learn_rate, double (*activation_function)(double, int)) {
    for (int i = 0; i < layer->size; i++) {
        layer->neurons[i].delta = 0.0;
        for (int j = 0; j < next_layer->size; j++) {
            double weight_next_neuron = next_layer->neurons[j].weights[i];
            double delta_next_neuron = next_layer->neurons[j].delta;
            layer->neurons[i].delta += weight_next_neuron * delta_next_neuron * activation_function(layer->neurons[i].weighted_input, 1);
        }

        for (int k = 0; k < layer->neurons[i].num_weights; k++) {
            double input = previous_layer->neurons[k].output;
            layer->neurons[i].weights[k] -= layer->neurons[i].delta * input * learn_rate;
        }

        layer->neurons[i].bias -= layer->neurons[i].delta * learn_rate;
    }
}

void learn(neural_network_t *nn, double learn_rate, data_t *training_data, int num_data) {
    for (int i = 0; i < num_data; i++) {
        layer_learn_output(nn, (nn->num_hidden_layer == 0) ? &nn->input_layer : &nn->hidden_layers[nn->num_hidden_layer - 1], &nn->output_layer, learn_rate, &training_data[i], nn->activation_function);
        for (int j = nn->num_hidden_layer - 1; j >= 0; j--) {
            layer_learn_intermediate((j == 0) ? &nn->input_layer : &nn->hidden_layers[j - 1], &nn->hidden_layers[j], (j == nn->num_hidden_layer - 1) ? &nn->output_layer : &nn->hidden_layers[j + 1], learn_rate, nn->activation_function);
        }
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

data_t data_from_image(char *path, float angle, float scale, int offset_x, int offset_y, float noise_factor, float probability) {
    data_t data;

    int width, height, channels;
    unsigned char *image = stbi_load(path, &width, &height, &channels, STBI_grey);
    if (image == NULL) {
        fprintf(stderr, "Failed to load image: %s\n", path);
        return data;
    }

    if (width != IMAGE_SIZE || height != IMAGE_SIZE || channels != 1) {
        fprintf(stderr, "Invalid image dimensions or channels: %s\n", path);
        stbi_image_free(image);
        return data;
    }

    unsigned char *scaled_image = scale_image(image, width, height, scale);
    unsigned char *rotated_image = rotate_image(scaled_image, width, height, angle);
    unsigned char *offset_image = add_offset(rotated_image, width, height, offset_x, offset_y);
    unsigned char *noisy_image = add_noise(offset_image, width, height, noise_factor, probability);

    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        data.inputs[i] = (double)noisy_image[i] / 255.0;
    }

    char *filename = strrchr(path, '/');
    if (filename == NULL) {
        filename = path;
    } else {
        filename--;
    }
    data.expected = atoi(filename);

    stbi_image_free(image);
    free(scaled_image);
    free(rotated_image);
    free(offset_image);
    free(noisy_image);
    return data;
}

float random_float(float min, float max) { return (float)rand() / RAND_MAX * (max - min) + min; }

data_t *populate_data_set(int *num_data, int max_each_digit, int *current_pos) {
    *num_data = 0;
    data_t *data_set = malloc(sizeof(data_t));
    int old_current = *current_pos;
    struct dirent *entry;
    int count = 0;
    for (int i = 0; i < 10; i++) {
        char subdirectory[30];
        sprintf(subdirectory, "data/train/%d", i);
        DIR *dir = opendir(subdirectory);
        if (dir == NULL) {
            fprintf(stderr, "Failed to open directory: %s\n", subdirectory);
            exit(1);
        }
        count = 0;
        while ((entry = readdir(dir)) != NULL && count - old_current < max_each_digit) {
            if (count < old_current) {
                count++;
                continue;
            }
            if (entry->d_type == DT_REG) {
                count++;
                char filepath[256];
                sprintf(filepath, "%s/%s", subdirectory, entry->d_name);
                data_t new_data = data_from_image(filepath, random_float(-5, 5), random_float(0.95, 1.05), random_float(-3, 3), random_float(-3, 3), 0.05, 0.05);
                *num_data += 1;
                data_set = realloc(data_set, sizeof(data_t) * (*num_data));
                data_set[*num_data - 1] = new_data;
            }
        }
        closedir(dir);
    }
    *current_pos += count - old_current;
    if (entry == NULL) {
        // no more files in this directory
        *current_pos = 0;
    }
    return data_set;
}

void save_network(const char *filename, neural_network_t *network) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file for writing\n");
        return;
    }

    fwrite(&(network->input_layer.size), sizeof(int), 1, file);
    fwrite(&(network->num_hidden_layer), sizeof(int), 1, file);

    for (int i = 0; i < network->num_hidden_layer; i++) {
        fwrite(&(network->hidden_layers[i].size), sizeof(int), 1, file);
        for (int j = 0; j < network->hidden_layers[i].size; j++) {
            fwrite(network->hidden_layers[i].neurons[j].weights, sizeof(double), network->hidden_layers[i].neurons[j].num_weights, file);
            fwrite(&(network->hidden_layers[i].neurons[j].bias), sizeof(double), 1, file);
        }
    }

    // Output layer
    fwrite(&(network->output_layer.size), sizeof(int), 1, file);
    for (int i = 0; i < network->output_layer.size; i++) {
        fwrite(network->output_layer.neurons[i].weights, sizeof(double), network->output_layer.neurons[i].num_weights, file);
        fwrite(&(network->output_layer.neurons[i].bias), sizeof(double), 1, file);
    }

    fclose(file);
}

void load_network(const char *filename, neural_network_t *network) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file for reading\n");
        return;
    }

    int check_val = 0;
    fread(&check_val, sizeof(int), 1, file);
    if (check_val != network->input_layer.size) {
        printf("Number of input layer not compatiable with save file, read: %d\n", check_val);
        return;
    }
    fread(&check_val, sizeof(int), 1, file);

    if (check_val != network->num_hidden_layer) {
        printf("Number of hidden layer not compatable with save file, read: %d\n", check_val);
        return;
    }
    for (int i = 0; i < network->num_hidden_layer; i++) {
        check_val = 0;
        fread(&check_val, sizeof(int), 1, file);
        if (check_val != network->hidden_layers[i].size) {
            printf("Number of hidden layer neuron not compatable with save file, read: %d\n", check_val);
            return;
        }
        for (int j = 0; j < network->hidden_layers[i].size; j++) {
            fread(network->hidden_layers[i].neurons[j].weights, sizeof(double), network->hidden_layers[i].neurons[j].num_weights, file);
            fread(&(network->hidden_layers[i].neurons[j].bias), sizeof(double), 1, file);
        }
    }

    // Output layer
    check_val = 0;
    fread(&check_val, sizeof(int), 1, file);
    if (check_val != network->output_layer.size) {
        printf("Number of output layer neuron not compatable with save file, read: %d\n", check_val);
        return;
    }
    for (int i = 0; i < network->output_layer.size; i++) {
        fread(network->output_layer.neurons[i].weights, sizeof(double), network->output_layer.neurons[i].num_weights, file);
        fread(&(network->output_layer.neurons[i].bias), sizeof(double), 1, file);
    }
    fclose(file);
}

double test_network_percent(neural_network_t *nn) {
    int tested = 0;
    int correct = 0;
    for (int i = 0; i < 10; i++) {
        char subdirectory[30];
        sprintf(subdirectory, "data/test/%d", i);
        DIR *dir = opendir(subdirectory);
        struct dirent *entry;
        if (dir == NULL) {
            fprintf(stderr, "Failed to open directory: %s\n", subdirectory);
            exit(1);
        }
        while ((entry = readdir(dir)) != NULL) {
            if (entry->d_type == DT_REG) {
                tested++;
                char filepath[256];
                sprintf(filepath, "%s/%s", subdirectory, entry->d_name);
                data_t test_data = data_from_image(filepath, 0, 1, 0, 0, 0, 0);
                add_inputs(nn, test_data.inputs);
                compute_network(nn);
                int max = 0;
                for (int i = 1; i < 10; i++) {
                    if (output_neuron_percent_activate(nn, i) > output_neuron_percent_activate(nn, 0)) {
                        max = i;
                    }
                }

                if (max == test_data.expected) {
                    correct++;
                }
            }
        }
        closedir(dir);
    }
    return (double)correct * 100.0 / (double)tested;
}

void train(neural_network_t *network, double learn_rate, int *num_data, int max_each, int learn_amount, int epoch_amount) {
    int current_pos = 0;
    data_t *training_data = populate_data_set(num_data, max_each, &current_pos);
    clock_t start_time = clock();
    for (int i = 0; i <= learn_amount; i++) {
        if (i % epoch_amount == 0 && i != 0) {
            free(training_data);
            training_data = populate_data_set(num_data, max_each, &current_pos);
            double new_cost = cost(network, training_data, *num_data);
            clock_t elapsed_ms = clock() - start_time;
            double elapsed_s = (double)elapsed_ms / CLOCKS_PER_SEC;
            double speed = (double)*num_data / elapsed_s * (double)epoch_amount;
            printf("Epoch learned %d, cost: %f, elapsed time: %.2fs, speed: %.2f Data/s \n", i, new_cost, elapsed_s, speed);
            start_time = clock();
        }
        learn(network, learn_rate, training_data, *num_data);
    }
    free(training_data);
}

// TODO:
// gpu parallelization

int main() {
    srand(time(NULL));
    int num_input = IMAGE_SIZE * IMAGE_SIZE;
    int num_hidden_layer = 2;
    int *hidden_layer_sizes = malloc(num_hidden_layer * sizeof(int));
    hidden_layer_sizes[0] = 100;
    hidden_layer_sizes[1] = 16;
    int output_layer_size = 10;
    double (*activation_function)(double, int) = &sigmoid;

    neural_network_t network;
    initialise_neural_network(&network, num_hidden_layer, hidden_layer_sizes, output_layer_size, num_input, activation_function);

    int num_data = 0;

    // Parameters
    int max_each = 10;
    double learn_rate = 0.03;
    int learn_amount = 1000;
    int epoch_amount = 64;

    char cmd[100];
    FILE *fp;
    double user_input[IMAGE_SIZE * IMAGE_SIZE];
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
            train(&network, learn_rate, &num_data, max_each, learn_amount, epoch_amount);
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
                    fclose(fp);
                    break;
                }

                char quit_flag;
                fscanf(fp, "%s", &quit_flag);
                if (quit_flag == 'q') {
                    fclose(fp);
                    break;
                }

                double generic_double = 0.0;
                int count = 0;
                while (fscanf(fp, "%lf", &generic_double) == 1) {
                    if (count > IMAGE_SIZE * IMAGE_SIZE) {
                        printf("Warning parsing input\n");
                        break;
                    }
                    user_input[count++] = generic_double;
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
    free_neural_network(&network);
    if (num_hidden_layer > 0) {
        free(hidden_layer_sizes);
    }
    return 0;
}
