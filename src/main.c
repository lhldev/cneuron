#include <cblas.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef USE_THREADING
#include <pthread.h>
#endif

#include "cneuron/cneuron.h"

const size_t IMAGE_SIZE = 28;

float sigmoid(float val, bool is_deravative) {
    float result = 1.0f / (1.0f + expf(-val));
    if (is_deravative)
        return result * (1.0f - result);

    return result;
}

typedef struct {
    dataset *train_dataset;
    size_t batch_size;
} generator_args;

dataset *dataset_generator(generator_args *args) {
    dataset *batch_dataset = get_random_dataset_sample(args->train_dataset, args->batch_size);
    for (size_t i = 0; i < batch_dataset->length; i++) {
        int stochastic_choice = rand() % 4;
        float *data = &batch_dataset->all_inputs[i * batch_dataset->inputs_length];
        switch (stochastic_choice) {
            case 0:
                rotate_data(data, IMAGE_SIZE, IMAGE_SIZE, randf(10.0f, -5.0f));
                break;
            case 1:
                scale_data(data, IMAGE_SIZE, IMAGE_SIZE, randf(1.2f, -0.1f));
                break;
            case 2:
                offset_data(data, IMAGE_SIZE, IMAGE_SIZE, randf(6.0f, -3.0f), randf(6.0f, -3.0f));
                break;
            case 3:
                noise_data(data, IMAGE_SIZE * IMAGE_SIZE, 0.3f, 0.08f);
                break;
        }
    }

    return batch_dataset;
}

void train(neural_network *nn, dataset *restrict train_dataset, dataset *restrict test_dataset, float learn_rate, size_t batch_size, unsigned int batch_amount) {
#ifdef USE_THREADING
    pthread_t thread;
#endif
    generator_args args = (generator_args){.train_dataset = train_dataset, .batch_size = batch_size};
    clock_t start_time = clock();
    dataset *batch_dataset = dataset_generator(&args);
#ifdef USE_THREADING
    if (batch_amount > 1) {
        pthread_create(&thread, NULL, (void *(*)(void *))dataset_generator, &args);
    }
#endif
    for (unsigned long i = 0; i < batch_amount; i++) {
        if (i != 0) {
            float new_cost = cost(nn, test_dataset, 100);
            clock_t elapsed_ms = clock() - start_time;
            float elapsed_s = (float)elapsed_ms / CLOCKS_PER_SEC;
            float speed = (float)batch_size / elapsed_s;
            printf("Learned: %zu, Batch: %zu, cost: %f, elapsed time: %.4fs, speed: %.2f Data/s\n", i * batch_size, i, new_cost, elapsed_s, speed);
            start_time = clock();
            free(batch_dataset);
#ifdef USE_THREADING
            pthread_join(thread, (void **)&batch_dataset);
            if (i < batch_amount - 1) {
                pthread_create(&thread, NULL, (void *(*)(void *))dataset_generator, &args);
            }
#else
            batch_dataset = dataset_generator(&args);
#endif
        }

        mini_batch_gd(nn, learn_rate, batch_dataset);
    }
    // Last dataset not used
    free(batch_dataset);
}

dataset *get_mnist(bool is_test) {
    char dir[512];
    snprintf(dir, sizeof(dir), "data/mnist/%s", is_test ? "test" : "train");

    dataset **datasets = malloc(sizeof(dataset *) * 10);

    for (size_t i = 0; i <= 9; i++) {
        char filepath[518];
        snprintf(filepath, sizeof(filepath), "%s/%zu.dat", dir, i);
        dataset *read_dataset = get_dataset(filepath);
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
    size_t inputs_length = datasets[0]->inputs_length;

    dataset *mnist_dataset = alloc_dataset(total_length, inputs_length);

    size_t curr_count = 0;
    for (size_t i = 0; i < 10; i++) {
        memcpy(&mnist_dataset->all_inputs[curr_count * inputs_length], datasets[i]->all_inputs, sizeof(float) * inputs_length * datasets[i]->length);
        memcpy(&mnist_dataset->expected_indices[curr_count], datasets[i]->expected_indices, sizeof(size_t) * datasets[i]->length);
        curr_count += datasets[i]->length;
        free(datasets[i]);
    }

    if (curr_count != mnist_dataset->length)
        printf("Error reading all mnist data. Read: %zu, Expected: %zu\n", curr_count, mnist_dataset->length);

    free(datasets);

    return mnist_dataset;
}

int main(int argc, char **argv) {
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
#endif

    dataset *train_dataset = get_mnist(false);
    dataset *test_dataset = get_mnist(true);
    const size_t network_length = 3;
    const size_t layer_lengths[] = {100, 16, 10};

    neural_network *nn = get_neural_network(network_length, layer_lengths, train_dataset->inputs_length);

    // Parameters
    const float learn_rate = 0.2f;
    const size_t batch_size = 1000;
    const unsigned int batch_amount = 200000;

    char cmd[100];
    FILE *fp;
    float user_input[IMAGE_SIZE * IMAGE_SIZE];
    bool loop = true;
    while (loop) {
        if (argc > 1) {
            loop = false;
            cmd[0] = argv[1][0];
        } else {
            printf("cmd: ");
            if (scanf("%99s", cmd) != 1) {
                printf("Invalid input format. Please try again.\n");
                continue;
            }
        }

        if (cmd[0] == 'q') {
            break;
        } else if (cmd[0] == 's') {
            if (save_network("output/nn.dat", nn)) {
                printf("Neural network saved!\n");
            }
        } else if (cmd[0] == 'l') {
            if (load_network("output/nn.dat", nn)) {
                printf("Neural network loaded!\n");
            }
        } else if (cmd[0] == 't') {
            train(nn, train_dataset, test_dataset, learn_rate, batch_size, batch_amount);
            printf("Training completed. Trained for %d batches.\n", batch_amount);
        } else if (cmd[0] == 'T') {
            printf("Testing neural network...\n");
            printf("Network is %.2f%% correct!\n", test_network_percent(nn, test_dataset));
        } else if (cmd[0] == 'i') {
            printf("Enter your input in the window and press enter...\n");
            while (1) {
                int ret = system("python3 -W ignore input.py");
                if (ret != 0) {
                    fprintf(stderr, "Command to open window failed with exit code %d\n", ret);
                    break;
                }

                // Open the file for reading
                fp = fopen("output/grid_array.txt", "r");
                if (!fp) {
                    fprintf(stderr, "Error opening file: %s\n", strerror(errno));
                    break;
                }

                char quit_flag;
                if (fscanf(fp, " %c", &quit_flag) == 1) {
                    if (quit_flag == 'q') {
                        fclose(fp);
                        break;
                    }
                }

                float generic_float = 0.0;
                int count = 0;
                while (fscanf(fp, "%f", &generic_float) == 1) {
                    if ((size_t)count >= IMAGE_SIZE * IMAGE_SIZE) {
                        fprintf(stderr, "Error parsing input\n");
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
    free(train_dataset);
    free(test_dataset);
    free(nn);
    return 0;
}
