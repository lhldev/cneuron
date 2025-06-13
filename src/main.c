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
#include "rand.h"

const size_t IMAGE_SIZE = 28;

float sigmoid(float val, bool is_deravative) {
    float result = 1.0f / (1.0f + expf(-val));
    if (is_deravative)
        return result * (1.0f - result);

    return result;
}

float relu(float val, bool is_deravative) {
    if (is_deravative)
        return (val > 0.0f) ? 1.0f : 0.0f;

    return fmax(0.0f, val);
}

typedef struct {
    dataset *train_dataset;
    size_t batch_size;
} generator_args;

dataset *dataset_generator(generator_args *args) {
    dataset *batch_dataset = get_random_dataset_sample(args->train_dataset, args->batch_size);
    for (size_t i = 0; i < batch_dataset->length; i++) {
        data *data = batch_dataset->datas[i];
        rotate_data(data, IMAGE_SIZE, IMAGE_SIZE, randf(10.0f, -5.0f));
        scale_data(data, IMAGE_SIZE, IMAGE_SIZE, randf(1.2f, -0.1f));
        offset_data(data, IMAGE_SIZE, IMAGE_SIZE, randf(6.0f, -3.0f), randf(6.0f, -3.0f));
        noise_data(data, IMAGE_SIZE * IMAGE_SIZE, 0.3f, 0.08f);
    }
    return batch_dataset;
}

void train(neural_network *nn, dataset *train_dataset, dataset *test_dataset, float learn_rate, int batch_amount, int log_amount, size_t batch_size) {
#ifdef USE_THREADING
    pthread_t thread;
#endif
    generator_args args = (generator_args){.train_dataset = train_dataset, .batch_size = batch_size};
    clock_t start_time = clock();
    dataset *batch_dataset = dataset_generator(&args);
    for (int i = 0; i < batch_amount; i++) {
        if (i % log_amount == 0 && i != 0) {
            float new_cost = cost(nn, test_dataset, 100);
            clock_t elapsed_ms = clock() - start_time;
            float elapsed_s = (float)elapsed_ms / CLOCKS_PER_SEC;
            float speed = (float)log_amount * batch_size / elapsed_s;
            printf("Learned: %zu, cost: %f, elapsed time: %.2fs, speed: %.2f Data/s\n", i * batch_size, new_cost, elapsed_s, speed);
            start_time = clock();
        }

#ifdef USE_THREADING
        pthread_create(&thread, NULL, (void *(*)(void *))dataset_generator, &args);
        mini_batch_gd(nn, learn_rate, batch_dataset);
        free_dataset(batch_dataset);
        void *result = NULL;
        pthread_join(thread, &result);
        batch_dataset = (dataset *)result;
#else
        mini_batch_gd(nn, learn_rate, batch_dataset);
        free_dataset(batch_dataset);
        batch_dataset = dataset_generator(&args);
#endif
    }
    // Last dataset not used
    free_dataset(batch_dataset);
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

    dataset *mnist_dataset = malloc(sizeof(dataset));
    mnist_dataset->datas = malloc(sizeof(data *) * total_length);
    mnist_dataset->length = total_length;
    mnist_dataset->inputs_length = IMAGE_SIZE * IMAGE_SIZE;

    size_t curr_count = 0;
    for (size_t i = 0; i < 10; i++) {
        for (size_t j = 0; j < datasets[i]->length; j++) {
            mnist_dataset->datas[curr_count] = datasets[i]->datas[j];
            curr_count++;
        }

        free(datasets[i]->datas);
        free(datasets[i]);
    }

    if (curr_count != mnist_dataset->length)
        printf("Error reading all mnist data. Read: %zu, Expected: %zu\n", curr_count, mnist_dataset->length);

    free(datasets);

    return mnist_dataset;
}

int main(int argc, char **argv) {
    dataset *train_dataset = get_mnist(false);
    dataset *test_dataset = get_mnist(true);
    size_t network_length = 3;
    size_t *layer_lengths = malloc(sizeof(size_t) * network_length);
    layer_lengths[0] = 100;
    layer_lengths[1] = 16;
    layer_lengths[2] = 10;

    neural_network *nn = get_neural_network(network_length, layer_lengths, train_dataset->inputs_length, &sigmoid);

    // Parameters
    float learn_rate = 1.5f;
    size_t batch_size = 30;
    int learn_amount = 48000000;
    int batch_amount = learn_amount / batch_size;
    int log_amount = 200;  // Log once reached a number of batch

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
            train(nn, train_dataset, test_dataset, learn_rate, batch_amount, log_amount, batch_size);
            printf("Training completed. Trained for %d times.\n", learn_amount);
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
    free_dataset(train_dataset);
    free_dataset(test_dataset);
    free_neural_network(nn);
    free(layer_lengths);
    return 0;
}
