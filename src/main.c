#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "data/data.h"
#include "network/network.h"

#define IMAGE_SIZE 28

float sigmoid(float val, int is_deravative) {
    float result = 1.0f / (1.0f + expf(-val));
    if (is_deravative == 1) {
        return result * (1.0f - result);
    }
    return result;
}

float relu(float val, int is_deravative) {
    if (is_deravative) {
        return (val > 0.0f) ? 1.0f : 0.0f;
    }
    return fmax(0.0f, val);
}

void train(neural_network_t *nn, dataset_t *dataset, dataset_t *test_dataset, float learn_rate, int learn_amount, int log_amount) {
    clock_t start_time = clock();
    for (int i = 0; i < learn_amount; i++) {
        if (i % log_amount == 0 && i != 0) {
            float new_cost = cost(nn, test_dataset, 100);
            clock_t elapsed_ms = clock() - start_time;
            float elapsed_s = (float)elapsed_ms / CLOCKS_PER_SEC;
            float speed = (float)log_amount / elapsed_s;
            printf("Learned: %d, cost: %f, elapsed time: %.2fs, speed: %.2f Data/s\n", i, new_cost, elapsed_s, speed);
            start_time = clock();
        }
        data_t *data = get_data_copy(dataset->datas[rand() % dataset->length], IMAGE_SIZE * IMAGE_SIZE);
        rotate_data(data, IMAGE_SIZE, IMAGE_SIZE, random_float(-5.0f, 5.0f));
        scale_data(data, IMAGE_SIZE, IMAGE_SIZE, random_float(0.9f, 1.1f));
        offset_data(data, IMAGE_SIZE, IMAGE_SIZE, random_float(-3.0f, 3.0f), random_float(-3.0f, 3.0f));
        noise_data(data, IMAGE_SIZE * IMAGE_SIZE, 0.3f, 0.08f);
        learn(nn, learn_rate, data);
        free_data(data);
    }
}

dataset_t *get_mnist(int is_test) {
    char dir[512];
    sprintf(dir, "data/mnist/%s", is_test ? "test" : "train");

    dataset_t **datasets = malloc(sizeof(dataset_t *) * 10);

    for (size_t i = 0; i <= 9; i++) {
        char filepath[518];
        sprintf(filepath, "%s/%zu.dat", dir, i);
        dataset_t *read_dataset = get_dataset(filepath);
        if (!read_dataset) {
            printf("Failed to load mnist dataset for digit %zu\n", i);
            return NULL;
        }

        datasets[i] = read_dataset;
    }

    size_t total_length = 0;
    for (size_t i = 0; i < 10; i++) {
        total_length += datasets[i]->length;
    }

    dataset_t *dataset = malloc(sizeof(dataset_t));
    dataset->datas = malloc(sizeof(data_t *) * total_length);
    dataset->length = total_length;
    dataset->inputs_length = IMAGE_SIZE * IMAGE_SIZE;

    size_t curr_count = 0;
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < datasets[i]->length; j++) {
            dataset->datas[curr_count] = datasets[i]->datas[j];
            curr_count++;
        }

        free(datasets[i]->datas);
        free(datasets[i]);
    }

    if (curr_count != dataset->length) {
        printf("Error reading all mnist data. Read: %zu, Expected: %zu\n", curr_count, dataset->length);
    }

    free(datasets);

    return dataset;
}

int main() {
    srand(time(NULL));
    dataset_t *dataset = get_mnist(0);
    dataset_t *test_dataset = get_mnist(1);
    size_t network_length = 3;
    size_t *layer_lengths = malloc(sizeof(size_t) * network_length);
    layer_lengths[0] = 100;
    layer_lengths[1] = 16;
    layer_lengths[2] = 10;

    neural_network_t *nn = get_neural_network(network_length, layer_lengths, dataset->inputs_length, &sigmoid);

    // Parameters
    float learn_rate = 0.03f;
    int learn_amount = 40000;
    int log_amount = 2000;

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
            save_network("output/nn.dat", nn);
            printf("Neural network saved!\n");
        } else if (cmd[0] == 'l') {
            load_network("output/nn.dat", nn);
            printf("Neural network loaded!\n");
        } else if (cmd[0] == 't') {
            train(nn, dataset, test_dataset, learn_rate, learn_amount, log_amount);
            printf("Training completed. Trained for %d times.\n", learn_amount);
        } else if (cmd[0] == 'T') {
            printf("Testing neural network...\n");
            printf("Network is %.2f%% correct!\n", test_network_percent(nn, test_dataset));
        } else if (cmd[0] == 'i') {
            printf("Enter your input in the window and press enter...\n");
            while (1) {
                system("python3 -W ignore input.py");

                // Open the file for reading
                fp = fopen("output/grid_array.txt", "r");
                if (!fp) {
                    printf("Error opening file\n");
                    break;
                }

                char quit_flag;
                fscanf(fp, " %c", &quit_flag);
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
                    user_input[count++] = generic_float;
                }
                fclose(fp);

                compute_network(nn, user_input);
                print_activation_percentages(nn);
            }
        } else {
            printf(
                "Command not recognised. \nt - train \nT - test network\ni - insert input \nl - load \ns - save \nq - "
                "quit\n");
            continue;
        }
    }
    free_dataset(dataset);
    free_dataset(test_dataset);
    free_neural_network(nn);
    free(layer_lengths);
    return 0;
}
