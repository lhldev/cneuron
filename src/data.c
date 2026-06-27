#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cneuron/cneuron.h"

dataset *alloc_dataset(size_t dataset_length, size_t inputs_length) {
    dataset *new_dataset = malloc(sizeof(dataset) + (sizeof(size_t) + sizeof(float) * inputs_length) * dataset_length);
    if (!new_dataset) return NULL;
    new_dataset->expected_indices = (size_t *)(new_dataset + 1);
    new_dataset->all_inputs = (float *)(new_dataset->expected_indices + inputs_length);
    new_dataset->length = dataset_length;
    new_dataset->inputs_length = inputs_length;

    return new_dataset;
}

dataset *get_dataset(const char *filename) {
    assert(filename);

    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file '%s' for reading data set: %s\n", filename, strerror(errno));
        return NULL;
    }

    size_t dataset_length = 0;
    if (fread(&dataset_length, sizeof(uint64_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read dataset length from %s\n", filename);
        fclose(file);
        return NULL;
    }

    size_t inputs_length = 0;
    if (fread(&inputs_length, sizeof(uint64_t), 1, file) != 1) {
        fprintf(stderr, "Failed to read inputs_length from %s\n", filename);
        fclose(file);
        return NULL;
    }

    dataset *read_dataset = alloc_dataset(dataset_length, inputs_length);
    if (!read_dataset) {
        fprintf(stderr, "Failed to allocate for dataset %s: %s\n", filename, strerror(errno));
        fclose(file);
        return NULL;
    }

    for (size_t i = 0; i < dataset_length; i++) {
        float *read_data = &read_dataset->all_inputs[i * inputs_length];
        size_t read_inputs = fread(read_data, sizeof(float), read_dataset->inputs_length, file);
        if (read_inputs != read_dataset->inputs_length) {
            fprintf(stderr, "Invalid inputs_length from %s. Expected: %zu. But found: %zu\n", filename, read_dataset->inputs_length, read_inputs);
            free(read_dataset);
            fclose(file);
            return NULL;
        }

        if (fread(&(read_dataset->expected_indices[i]), sizeof(uint64_t), 1, file) != 1) {
            fprintf(stderr, "Failed to read expected_index from %s\n", filename);
            free(read_dataset);
            fclose(file);
            return NULL;
        }
    }

    fclose(file);

    return read_dataset;
}

dataset *get_random_dataset_sample(const dataset *source_dataset, size_t amount) {
    assert(source_dataset);
    size_t inputs_size = source_dataset->inputs_length;
    dataset *new_dataset = alloc_dataset(amount, inputs_size);
    if (!new_dataset) {
        return NULL;
    }
    for (size_t i = 0; i < amount; i++) {
        uint32_t randnum = randnum_u32(source_dataset->length, 0);
        float *random_data = &source_dataset->all_inputs[randnum * inputs_size];
        float *target_data = &new_dataset->all_inputs[i * inputs_size];
        memcpy(target_data, random_data, sizeof(float) * inputs_size);
        new_dataset->expected_indices[i] = source_dataset->expected_indices[randnum];
    }

    return new_dataset;
}

void rotate_data(float *data, int width, int height, float angle) {
    assert(width > 0 && height > 0);

    float rad = angle * M_PI / 180.0f;
    float cos_angle = cosf(rad);
    float sin_angle = sinf(rad);
    float *new_inputs = calloc(width * height, sizeof(float));
    if (!new_inputs) {
        return;
    }

    int center_x = floorf(width / 2.0f);
    int center_y = floorf(height / 2.0f);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int src_x = roundf((x - center_x) * cos_angle + (y - center_y) * sin_angle + center_x);
            int src_y = roundf((y - center_y) * cos_angle - (x - center_x) * sin_angle + center_y);

            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                new_inputs[y * width + x] = data[src_y * width + src_x];
            }
        }
    }

    memcpy(data, new_inputs, sizeof(float) * width * height);
    free(new_inputs);
}

void scale_data(float *data, int width, int height, float scale) {
    assert(width > 0 && height > 0);

    int scale_width = roundf(width * scale);
    int scale_height = roundf(height * scale);
    float *new_inputs = calloc(width * height, sizeof(float));
    if (!new_inputs) {
        return;
    }

    int offset_x = roundf((scale_width - width) / 2.0f);
    int offset_y = roundf((scale_height - height) / 2.0f);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int scaled_x = x + offset_x;
            int scaled_y = y + offset_y;

            int src_x = roundf(scaled_x / scale);
            int src_y = roundf(scaled_y / scale);

            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                new_inputs[y * width + x] = data[src_y * width + src_x];
            }
        }
    }

    memcpy(data, new_inputs, sizeof(float) * width * height);
    free(new_inputs);
}

void offset_data(float *data, int width, int height, float offset_x, float offset_y) {
    assert(width > 0 && height > 0);

    float *new_inputs = calloc(width * height, sizeof(float));
    if (!new_inputs) {
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float new_x = x - offset_x;
            float new_y = y - offset_y;

            int src_x = roundf(new_x);
            int src_y = roundf(new_y);
            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                new_inputs[y * width + x] = data[src_y * width + src_x];
            }
        }
    }
    memcpy(data, new_inputs, sizeof(float) * width * height);
    free(new_inputs);
}

void noise_data(float *data, size_t inputs_length, float noise_factor, float probability) {
    assert(inputs_length > 0);

    for (size_t i = 0; i < inputs_length; i++) {
        if (randf(1.0f, 0.0f) <= probability) {
            float noise = randf(noise_factor, 0);
            float new_value = data[i] + noise;

            data[i] = fmin(new_value, 1.0f);
        }
    }
}
