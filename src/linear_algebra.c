#include <assert.h>
#include <stdbool.h>

#include "cneuron/cneuron.h"

void vector_apply_activation(const float *a, float *b, size_t length, float (*activation_function)(float, bool), bool is_derivative) {
    assert(a && b && activation_function);
    for (size_t i = 0; i < length; i++) {
        b[i] = activation_function(a[i], is_derivative);
    }
}

void hadamard_product(const float *a, const float *b, float *c, size_t length) {
    assert(a && b && c);

    for (size_t i = 0; i < length; i++) {
        c[i] = a[i] * b[i];
    }
}
