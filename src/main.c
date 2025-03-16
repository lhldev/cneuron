#include "data/data.h"
#include "network/network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define IMAGE_SIZE 28

float sigmoid(float val, int is_deravative) {
    float result = 1.0f / (1.0f + exp(-val));
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

// TODO:
// gpu parallelization

int main() {
    srand(time(NULL));
    dataset_t *dataset = get_dataset("data/mnist/mnist_train.dat");
    dataset_t *test_dataset = get_dataset("data/mnist/mnist_test.dat");
    int num_hidden_layer = 2;
    int *hidden_layer_sizes = malloc(num_hidden_layer * sizeof(int));
    hidden_layer_sizes[0] = 100;
    hidden_layer_sizes[1] = 16;
    int output_layer_size = 10;

    neural_network_t network;
    initialise_neural_network(&network, num_hidden_layer, hidden_layer_sizes, output_layer_size, dataset->inputs_length, &sigmoid);

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
            save_network("output/nn.dat", &network);
            printf("Neural network saved!\n");
        } else if (cmd[0] == 'l') {
            load_network("output/nn.dat", &network);
            printf("Neural network loaded!\n");
        } else if (cmd[0] == 't') {
            train(&network, dataset, test_dataset, learn_rate, learn_amount, log_amount);
            printf("Training completed. Trained for %d times.\n", learn_amount);
        } else if (cmd[0] == 'T') {
            printf("Testing neural network...\n");
            printf("Network is %.2f%% correct!\n", test_network_percent(&network, test_dataset));
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
    free_dataset(dataset);
    free_dataset(test_dataset);
    free_neural_network(&network);
    if (num_hidden_layer > 0) {
        free(hidden_layer_sizes);
    }
    return 0;
}
