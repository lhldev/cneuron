#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cneuron/cneuron.h"

#define BACKGROUND_VALUE 0.0f

dataset *get_dataset(const char *filename) {
    assert(filename);

    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file '%s' for reading data set: %s\n", filename, strerror(errno));
        return NULL;
    }

    dataset *read_dataset = malloc(sizeof(dataset));
    if (!read_dataset) {
        fclose(file);
        return NULL;
    }

    if (fread(&read_dataset->length, sizeof(uint64_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read dataset length from %s\n", filename);
        free(read_dataset);
        fclose(file);
        return NULL;
    }

    read_dataset->datas = calloc(read_dataset->length, sizeof(data *));
    if (!read_dataset->datas) {
        free(read_dataset);
        fclose(file);
        return NULL;
    }

    if (fread(&read_dataset->inputs_length, sizeof(uint64_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read inputs_length from %s\n", filename);
        free(read_dataset);
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < read_dataset->length; i++) {
        data *read_data = malloc(sizeof(data));
        if (!read_data) {
            goto cleanup;
        }

        read_data->inputs = malloc(sizeof(float) * read_dataset->inputs_length);
        if (!read_data->inputs) {
            free(read_data);
            goto cleanup;
        }

        size_t read_inputs = fread(read_data->inputs, sizeof(float), read_dataset->inputs_length, file);
        if (read_inputs != read_dataset->inputs_length) {
            fprintf(stderr, "Invalid inputs_length from %s. Expected: %zu. But found: %zu\n", filename, read_dataset->inputs_length, read_inputs);
            free_data(read_data);
            goto cleanup;
        }

        if (fread(&(read_data->expected_index), sizeof(uint64_t), 1, file) != 1) {
            fprintf(stderr, "Failed to read expected_index from %s\n", filename);
            free_data(read_data);
            goto cleanup;
        }

        read_dataset->datas[i] = read_data;
    }

    fclose(file);

    return read_dataset;

cleanup:
    free_dataset(read_dataset);
    fclose(file);
    return NULL;
}

void free_dataset(dataset *dataset) {
    if (!dataset) {
        return;
    }

    for (size_t i = 0; i < dataset->length; i++) {
        free_data(dataset->datas[i]);
    }
    free(dataset->datas);
    free(dataset);
}

void free_data(data *data) {
    if (!data) {
        return;
    }

    free(data->inputs);
    free(data);
}

data *get_data_copy(const data *source_data, size_t inputs_length) {
    assert(source_data);
    assert(source_data->inputs);
    assert(inputs_length > 0);

    data *copy = malloc(sizeof(data));
    if (!copy) {
        return NULL;
    }

    copy->expected_index = source_data->expected_index;

    size_t inputs_size = sizeof(float) * inputs_length;
    copy->inputs = malloc(inputs_size);
    if (!copy->inputs) {
        free(copy);
        return NULL;
    }

    memcpy(copy->inputs, source_data->inputs, inputs_size);

    return copy;
}

void rotate_data(data *data, int width, int height, float angle) {
    assert(data);
    assert(data->inputs);
    assert(width > 0 && height > 0);

    float rad = angle * M_PI / 180.0f;
    float cos_angle = cos(rad);
    float sin_angle = sin(rad);
    float *new_inputs = malloc(sizeof(float) * width * height);
    if (!new_inputs) {
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int center_x = floor(width / 2.0f);
            int center_y = floor(height / 2.0f);
            int src_x = round((x - center_x) * cos_angle + (y - center_y) * sin_angle + center_x);
            int src_y = round((y - center_y) * cos_angle - (x - center_x) * sin_angle + center_y);

            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                new_inputs[y * width + x] = data->inputs[src_y * width + src_x];
            } else {
                new_inputs[y * width + x] = BACKGROUND_VALUE;
            }
        }
    }

    free(data->inputs);
    data->inputs = new_inputs;
}

void scale_data(data *data, int width, int height, float scale) {
    assert(data);
    assert(data->inputs);
    assert(width > 0 && height > 0);

    int scale_width = round(width * scale);
    int scale_height = round(height * scale);
    float *new_inputs = malloc(sizeof(float) * width * height);
    if (!new_inputs) {
        return;
    }

    int offset_x = round((scale_width - width) / 2.0f);
    int offset_y = round((scale_height - height) / 2.0f);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int scaled_x = x + offset_x;
            int scaled_y = y + offset_y;

            int src_x = round(scaled_x / scale);
            int src_y = round(scaled_y / scale);

            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                new_inputs[y * width + x] = data->inputs[src_y * width + src_x];
            } else {
                new_inputs[y * width + x] = BACKGROUND_VALUE;
            }
        }
    }

    free(data->inputs);
    data->inputs = new_inputs;
}

void offset_data(data *data, int width, int height, float offset_x, float offset_y) {
    assert(data);
    assert(data->inputs);
    assert(width > 0 && height > 0);

    float *new_inputs = malloc(sizeof(float) * width * height);
    if (!new_inputs) {
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float new_x = x - offset_x;
            float new_y = y - offset_y;

            int src_x = round(new_x);
            int src_y = round(new_y);
            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                new_inputs[y * width + x] = data->inputs[src_y * width + src_x];
            } else {
                new_inputs[y * width + x] = BACKGROUND_VALUE;
            }
        }
    }
    free(data->inputs);
    data->inputs = new_inputs;
}

void noise_data(data *data, size_t inputs_length, float noise_factor, float probability) {
    assert(data);
    assert(data->inputs);
    assert(inputs_length > 0);

    for (size_t i = 0; i < inputs_length; i++) {
        float random_value = rand() / (float)RAND_MAX;
        if (random_value <= probability) {
            float noise = (rand() / (float)RAND_MAX * noise_factor);
            float new_value = data->inputs[i] + noise;

            data->inputs[i] = fmin(new_value, 1.0f);
        }
    }
}

float output_expected(size_t index, const data *data) {
    assert(data);

    if (index == data->expected_index) {
        return 1.0f;
    } else {
        return 0.0f;
    }
}
