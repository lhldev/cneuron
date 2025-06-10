#include <assert.h>
#include <stdbool.h>

#include "cneuron/cneuron.h"

void matrix_multiply(const float *a, const float *b, float *c, size_t rows_a, size_t cols_a, size_t cols_b) {
    assert(a);
    assert(b);
    assert(c);

    for (size_t col = 0; col < cols_b; ++col) {
        for (size_t row = 0; row < rows_a; ++row) {
            float sum = 0.0f;
            for (size_t k = 0; k < cols_a; ++k) {
                sum += a[k * rows_a + row] * b[col * cols_a + k];
            }
            c[col * rows_a + row] = sum;
        }
    }
}

void vector_apply_activation(const float *a, float *b, size_t length, float (*activation_function)(float, bool)) {
    assert(a);
    assert(b);
    assert(activation_function);

    for (size_t i = 0; i < length; i++) {
        b[i] = activation_function(a[i], false);
    }
}

void vector_add(const float *a, const float *b, float *c, size_t length) {
    assert(a);
    assert(b);
    assert(c);

    for (size_t i = 0; i < length; i++) {
        c[i] = a[i] + b[i];
    }
}
