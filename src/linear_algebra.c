#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "cneuron/cneuron.h"

void vector_apply_activation(const float *a, float *b, size_t length) {
    assert(a && b);
    for (size_t i = 0; i < length; i++) {
        b[i] = 1.0f / (1.0f + expf(-a[i]));
    }
}

void vector_apply_d_activation(const float *a, float *b, size_t length) {
    assert(a && b);
    for (size_t i = 0; i < length; i++) {
        float result = 1.0f / (1.0f + expf(-a[i]));
        b[i] = result * (1.0f - result);
    }
}

void hadamard_product(const float *restrict a, const float *restrict b, float *restrict c, size_t length) {
    assert(a && b && c);

    for (size_t i = 0; i < length; i++) {
        c[i] = a[i] * b[i];
    }
}
