#include <gtest/gtest.h>
#include <math.h>

extern "C" {
#include "cneuron/cneuron.h"
}

#include "test_utils.h"

TEST(LinearAlgebraTest, VectorApplyActivation) {
    size_t length = 3;
    float *a = (float *)malloc(sizeof(float) * length);
    float *b = (float *)malloc(sizeof(float) * length);

    a[0] = 1.0f;
    a[1] = -1.2f;
    a[2] = -0.2f;

    vector_apply_activation(a, b, length, sigmoid, false);

    for (size_t i = 0; i < length; i++) {
        ASSERT_FLOAT_EQ(b[i], sigmoid(a[i], false));
    }

    free(a);
    free(b);
}

TEST(LinearAlgebraTest, HadamardProduct) {
    size_t length = 3;
    float *a = (float *)malloc(sizeof(float) * length);
    float *b = (float *)malloc(sizeof(float) * length);

    a[0] = 1.0f;
    a[1] = -1.2f;
    a[2] = -0.2f;

    b[0] = 1.2f;
    b[1] = -2.2f;
    b[2] = -0.4f;

    hadamard_product(a, b, b, length);

    ASSERT_FLOAT_EQ(b[0], 1.2f);
    ASSERT_FLOAT_EQ(b[1], 2.64f);
    ASSERT_FLOAT_EQ(b[2], 0.08f);

    free(a);
    free(b);
}
