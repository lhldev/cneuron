#include <gtest/gtest.h>

#include "test_utils.h"

extern "C" {
#include "cneuron/cneuron.h"
}

#include <math.h>

TEST(LinearAlgebraTest, MatrixMultiply) {
    size_t rows_a = 3;
    size_t cols_a = 1;
    size_t rows_b = 1;
    size_t cols_b = 3;
    float *a = (float *)malloc(sizeof(float) * rows_a * cols_a);
    float *b = (float *)malloc(sizeof(float) * rows_b * cols_b);

    a[0] = 1.0f;
    a[1] = 2.0f;
    a[2] = 3.0f;
    b[0] = 4.0f;
    b[1] = 5.0f;
    b[2] = 6.0f;

    float *c = (float *)malloc(sizeof(float) * rows_a * cols_b);
    matrix_vector_multiply(a, b, c, rows_a, cols_a);
    ASSERT_FLOAT_EQ(c[0], 4.0f);
    ASSERT_FLOAT_EQ(c[2], 12.0f)
    ASSERT_FLOAT_EQ(c[4], 10.0f);
    ASSERT_FLOAT_EQ(c[7], 12.0f);

    free(a);
    free(b);
    free(c);
}

TEST(LinearAlgebraTest, VectorApplyActivation) {
    size_t length = 3;
    float *a = (float *)malloc(sizeof(float) * length);
    float *b = (float *)malloc(sizeof(float) * length);

    a[0] = 1.0;
    a[1] = -1.2;
    a[2] = -0.2;

    vector_apply_activation(a, b, length, sigmoid);

    for (size_t i = 0; i < length; i++) {
        ASSERT_FLOAT_EQ(b[i], sigmoid(a[i], false));
    }

    free(a);
    free(b);
}

TEST(LinearAlgebraTest, VectorAdd) {
    size_t length = 3;
    float *a = (float *)malloc(sizeof(float) * length);
    float *b = (float *)malloc(sizeof(float) * length);
    float *c = (float *)malloc(sizeof(float) * length);

    a[0] = 1.0;
    a[1] = -1.2;
    a[1] = -0.2;

    b[0] = 0.2;
    b[1] = -0.2;
    b[2] = 0.6;

    vector_add(a, b, c, length);

    for (size_t i = 0; i < length; i++) {
        ASSERT_FLOAT_EQ(c[i], a[i] + b[i]);
    }

    free(a);
    free(b);
    free(c);
}
