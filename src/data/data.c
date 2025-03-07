#include "data.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

data_t **get_dataset(const char *filename, unsigned int *dataset_length, unsigned int *inputs_length) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening %s for reading data set\n", filename);
        return NULL;
    }

    fread(dataset_length, sizeof(unsigned int), 1, file);
    data_t **dataset = malloc(sizeof(data_t*) * *dataset_length);

    fread(inputs_length, sizeof(unsigned int), 1, file);
    for (unsigned int i = 0; i < *dataset_length; i++) {
        data_t *data = malloc(sizeof(data_t));
        data->inputs = malloc(sizeof(float) * *inputs_length);
        fread(data->inputs, sizeof(float), *inputs_length, file);
        fread(&(data->neuron_index), sizeof(unsigned int), 1, file);

        dataset[i] = data;
    }

    fclose(file);

    return dataset;
}

void free_dataset(data_t **dataset, unsigned int dataset_length) {
    for (unsigned int i = 0; i < dataset_length; i++) {
        free(dataset[i]->inputs);
    }
    free(dataset);
}
