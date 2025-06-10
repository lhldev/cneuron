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
    dataset *test_dataset = (dataset *)malloc(sizeof(dataset));
    test_dataset->length = 4;
    data **datas = (data **)malloc(sizeof(data *) * test_dataset->length);
    test_dataset->datas = datas;
    test_dataset->inputs_length = 2;

    for (size_t i = 0; i < test_dataset->length; i++) {
        test_dataset->datas[i] = (data *)malloc(sizeof(data));
        test_dataset->datas[i]->inputs = (float *)malloc(sizeof(float) * test_dataset->inputs_length);
    }

    // XOR gate
    test_dataset->datas[0]->inputs[0] = 1.0f;
    test_dataset->datas[0]->inputs[1] = 1.0f;
    test_dataset->datas[0]->expected_index = 0;

    test_dataset->datas[1]->inputs[0] = 0.0f;
    test_dataset->datas[1]->inputs[1] = 0.0f;
    test_dataset->datas[1]->expected_index = 0;

    test_dataset->datas[2]->inputs[0] = 0.0f;
    test_dataset->datas[2]->inputs[1] = 1.0f;
    test_dataset->datas[2]->expected_index = 1;

    test_dataset->datas[3]->inputs[0] = 1.0f;
    test_dataset->datas[3]->inputs[1] = 0.0f;
    test_dataset->datas[3]->expected_index = 1;

    return test_dataset;
}

dataset *get_or() {
    // Create data
    dataset *test_dataset = (dataset *)malloc(sizeof(dataset));
    test_dataset->length = 4;
    data **datas = (data **)malloc(sizeof(data *) * test_dataset->length);
    test_dataset->datas = datas;
    test_dataset->inputs_length = 2;

    for (size_t i = 0; i < test_dataset->length; i++) {
        test_dataset->datas[i] = (data *)malloc(sizeof(data));
        test_dataset->datas[i]->inputs = (float *)malloc(sizeof(float) * test_dataset->inputs_length);
    }

    // OR gate
    test_dataset->datas[1]->inputs[0] = 0.0f;
    test_dataset->datas[1]->inputs[1] = 0.0f;
    test_dataset->datas[1]->expected_index = 0;

    test_dataset->datas[0]->inputs[0] = 1.0f;
    test_dataset->datas[0]->inputs[1] = 1.0f;
    test_dataset->datas[0]->expected_index = 1;

    test_dataset->datas[2]->inputs[0] = 0.0f;
    test_dataset->datas[2]->inputs[1] = 1.0f;
    test_dataset->datas[2]->expected_index = 1;

    test_dataset->datas[3]->inputs[0] = 1.0f;
    test_dataset->datas[3]->inputs[1] = 0.0f;
    test_dataset->datas[3]->expected_index = 1;

    return test_dataset;
}
