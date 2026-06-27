#include "test_utils.h"

#include <math.h>

float sigmoid(float val, bool is_deravative) {
    float result = 1.0f / (1.0f + expf(-val));
    if (is_deravative == 1) {
        return result * (1.0f - result);
    }
    return result;
}

dataset *get_xor() {
    // Create data
    size_t dataset_length = 4;
    size_t inputs_length = 2;
    dataset *test_dataset = alloc_dataset(dataset_length, inputs_length);

    // XOR gate
    test_dataset->all_inputs[0] = 1.0f;
    test_dataset->all_inputs[1] = 1.0f;
    test_dataset->expected_indices[0] = 0;

    test_dataset->all_inputs[2] = 0.0f;
    test_dataset->all_inputs[3] = 0.0f;
    test_dataset->expected_indices[1] = 0;

    test_dataset->all_inputs[4] = 0.0f;
    test_dataset->all_inputs[5] = 1.0f;
    test_dataset->expected_indices[2] = 1;

    test_dataset->all_inputs[6] = 1.0f;
    test_dataset->all_inputs[7] = 0.0f;
    test_dataset->expected_indices[3] = 1;

    return test_dataset;
}

dataset *get_or() {
    // Create data
    size_t dataset_length = 4;
    size_t inputs_length = 2;
    dataset *test_dataset = alloc_dataset(dataset_length, inputs_length);

    // OR gate
    test_dataset->all_inputs[0] = 0.0f;
    test_dataset->all_inputs[1] = 0.0f;
    test_dataset->expected_indices[0] = 0;

    test_dataset->all_inputs[2] = 1.0f;
    test_dataset->all_inputs[3] = 1.0f;
    test_dataset->expected_indices[1] = 1;

    test_dataset->all_inputs[4] = 0.0f;
    test_dataset->all_inputs[5] = 1.0f;
    test_dataset->expected_indices[2] = 1;

    test_dataset->all_inputs[6] = 1.0f;
    test_dataset->all_inputs[7] = 0.0f;
    test_dataset->expected_indices[3] = 1;

    return test_dataset;
}
