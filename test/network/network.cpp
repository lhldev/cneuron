#include <gtest/gtest.h>

extern "C" {
    #include "network/network.h"
}

#include <math.h>

float sigmoid(float val, int is_deravative) {
    float result = 1.0f / (1.0f + exp(-val));
    if (is_deravative == 1) {
        return result * (1.0f - result);
    }
    return result;
}

TEST(NetworkTest, RandomFloat) {
    float test = random_float(0.0f, 1.0f);
    bool same = true;
    for (int i = 0; i < 10; i++) {
        if (test != random_float(0.0f, 1.0f)) {
            same = false;
            break;
        }
    }
    ASSERT_FALSE(same);
}

TEST(NetworkTest, MatrixMultiply) {
    float *a = (float*)malloc(sizeof(float) * 3);
    float *b = (float*)malloc(sizeof(float) * 3);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;

    float *c = (float*)malloc(sizeof(float) * 9);
    matrix_multiply(a, b, c, 3, 1, 3);
    ASSERT_FLOAT_EQ(c[0], 4.0f);
    ASSERT_FLOAT_EQ(c[2], 12.0f);
    ASSERT_FLOAT_EQ(c[4], 10.0f);
    ASSERT_FLOAT_EQ(c[7], 12.0f);

    free(a);
    free(b);
    free(c);
}

TEST(NetworkTest, GetLayer) {
    size_t layer_length = 3;
    layer_t *layer = get_layer(layer_length, 5);

    ASSERT_NE(layer, nullptr);
    ASSERT_NE(layer->delta, nullptr);
    ASSERT_NE(layer->weighted_input, nullptr);
    ASSERT_NE(layer->weights, nullptr);
    ASSERT_NE(layer->bias, nullptr);
    ASSERT_NE(layer->output, nullptr);
    
    ASSERT_EQ(layer->length, layer_length);

    free_layer(layer);
}

TEST(NetworkTest, GetNeuralNetwork) {
    size_t layer_length = 3;
    size_t *layer_lengths = (size_t*)malloc(sizeof(size_t) * layer_length);
    layer_lengths[0] = 2;
    layer_lengths[1] = 3;
    layer_lengths[2] = 4;
    size_t inputs_length = 2;
    neural_network_t *nn = get_neural_network(layer_length, layer_lengths, inputs_length, &sigmoid);

    ASSERT_NE(nn, nullptr);
    ASSERT_EQ(nn->length, layer_length);
    ASSERT_EQ(nn->inputs_length, inputs_length);
    ASSERT_EQ(nn->activation_function, &sigmoid);
    ASSERT_NE(nn->layers, nullptr);
    for (size_t i = 0; i < layer_length; i++) {
        ASSERT_NE(nn->layers[i], nullptr);
        ASSERT_EQ(nn->layers[i]->length, layer_lengths[i]);
        ASSERT_EQ(nn->layers[i]->prev_layer, (i == 0) ? nullptr : nn->layers[i - 1]);
        ASSERT_EQ(nn->layers[i]->next_layer, (i == layer_length - 1) ? nullptr : nn->layers[i + 1]);
    }

    free_neural_network(nn);
    free(layer_lengths);
}

TEST(NetworkTest, FreeDataset) {
    size_t layer_length = 1;
    size_t *layer_lengths = (size_t*)malloc(sizeof(size_t) * layer_length);
    layer_lengths[0] = 2;
    neural_network_t *nn = get_neural_network(layer_length, layer_lengths, 2, nullptr);

    free_neural_network(nn);
    // No crash
    free(layer_lengths);
}

TEST(NetworkTest, FreeLayer) {
    size_t layer_length = 2;
    layer_t *layer = get_layer(layer_length, 3);

    free_layer(layer);
    // No crash
}

TEST(NetworkTest, ComputeNetwork) {
    size_t layer_length = 1;
    size_t inputs_length = 1;
    size_t *layer_lengths = (size_t*)malloc(sizeof(size_t) * layer_length);
    layer_lengths[0] = 1;
    neural_network_t *nn = get_neural_network(layer_length, layer_lengths, inputs_length, &sigmoid);

    float *inputs = (float*)malloc(sizeof(float) * inputs_length);
    inputs[0] = 0.2f;

    nn->layers[0]->weights[0] = 0.5f;
    nn->layers[0]->bias[0] = 0.3f;

    compute_network(nn, inputs);

    ASSERT_FLOAT_EQ(nn->layers[0]->output[0], 0.59868766f);

    free(inputs);
    free_neural_network(nn);
    free(layer_lengths);
}
