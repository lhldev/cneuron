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
    test_dataset->datas[0].inputs[0] = 1.0f;
    test_dataset->datas[0].inputs[1] = 1.0f;
    test_dataset->datas[0].expected_index = 0;

    test_dataset->datas[1].inputs[0] = 0.0f;
    test_dataset->datas[1].inputs[1] = 0.0f;
    test_dataset->datas[1].expected_index = 0;

    test_dataset->datas[2].inputs[0] = 0.0f;
    test_dataset->datas[2].inputs[1] = 1.0f;
    test_dataset->datas[2].expected_index = 1;

    test_dataset->datas[3].inputs[0] = 1.0f;
    test_dataset->datas[3].inputs[1] = 0.0f;
    test_dataset->datas[3].expected_index = 1;

    return test_dataset;
}

dataset *get_or() {
    // Create data
    size_t dataset_length = 4;
    size_t inputs_length = 2;
    dataset *test_dataset = alloc_dataset(dataset_length, inputs_length);

    // OR gate
    test_dataset->datas[1].inputs[0] = 0.0f;
    test_dataset->datas[1].inputs[1] = 0.0f;
    test_dataset->datas[1].expected_index = 0;

    test_dataset->datas[0].inputs[0] = 1.0f;
    test_dataset->datas[0].inputs[1] = 1.0f;
    test_dataset->datas[0].expected_index = 1;

    test_dataset->datas[2].inputs[0] = 0.0f;
    test_dataset->datas[2].inputs[1] = 1.0f;
    test_dataset->datas[2].expected_index = 1;

    test_dataset->datas[3].inputs[0] = 1.0f;
    test_dataset->datas[3].inputs[1] = 0.0f;
    test_dataset->datas[3].expected_index = 1;

    return test_dataset;
}
