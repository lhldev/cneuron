#pragma once

typedef struct {
    float *inputs;
    unsigned int neuron_index;
} data_t;

typedef struct {
    unsigned int length;
    unsigned int inputs_length;
    data_t **datas;
} dataset_t;

dataset_t *get_dataset(const char *filename);

void free_dataset(dataset_t* dataset);
void free_data(data_t* data);

data_t *get_data_copy(data_t* data, unsigned int inputs_length);

void rotate_data(data_t *data, int width, int height, float angle);
void scale_data(data_t *data, int width, int height, float scale);
void offset_data(data_t *data, int width, int height, float offset_x, float offset_y);
void noise_data(data_t *data, unsigned int inputs_length, float noise_factor, float probability);
