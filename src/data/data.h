#pragma once

typedef struct {
    float *inputs;
    unsigned int neuron_index;
} data_t;

data_t **get_dataset(const char *filename, unsigned int *dataset_length ,unsigned int *inputs_length);

void free_dataset(data_t** dataset, unsigned int dataset_length);
