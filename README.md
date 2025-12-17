# C Neural Network 🧠

## Prerequisites
Ensure that a BLAS (Basic Linear Algebra Subprograms) library is installed. This project relies on the C interface to BLAS (cblas.h), which is provided by most major BLAS distributions, including:
- Intel MKL
- OpenBLAS

## For optimal performance, compile the project using the following cmake command:
```
cmake -S . -B build -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_BUILD_TYPE=Release -DENABLE_TESTING=OFF
```
## Minimal code example (training on xor)
```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cneuron/cneuron.h"

float sigmoid(float val, bool is_deravative) {
    float result = 1.0f / (1.0f + expf(-val));
    if (is_deravative == 1) {
        return result * (1.0f - result);
    }
    return result;
}

int main() {
    // Create XOR dataset
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

    // Create network
    size_t layer_length = 2;
    size_t *layer_lengths = (size_t *)malloc(sizeof(size_t) * layer_length);
    layer_lengths[0] = 4;
    layer_lengths[1] = 2;
    neural_network *nn = get_neural_network(layer_length, layer_lengths, test_dataset->inputs_length, &sigmoid);

    for (size_t i = 0; i < 500000; i++) {
        for (size_t j = 0; j < test_dataset->length; j++) {
            stochastic_gd(nn, 0.001f, &test_dataset->datas[randnum_u32(test_dataset->length, 0)]);
        }
        if (i % 100000 == 0) {
            printf("Stochastic Multi layer learn cost: %f\n", cost(nn, test_dataset, test_dataset->length));
        }
    }

    free(nn);
    free(layer_lengths);
    free(test_dataset);
    return 0;
}
```
## Benchmark - Highest average recorded
- Intel Core i5 9th Gen: ~150,000 Data/s
- Intel Core Ultra 5: ~250,000 Data/s

## This project utilizes the **MNIST dataset**
Information regarding its license (Creative Commons Attribution-ShareAlike 3.0) and attribution can be found in the [data/mnist/MNIST_Copyright.md](data/mnist/MNIST_Copyright.md) file
