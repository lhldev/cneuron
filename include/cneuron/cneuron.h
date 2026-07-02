#ifndef CNEURON_H
#define CNEURON_H

#ifdef __cplusplus
#define restrict __restrict
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Represents a dataset that contains multiple data elements.
 */
typedef struct {
    size_t length;            /**< Number of data elements in the dataset. */
    size_t inputs_length;     /**< Number of input values per data element. */
    size_t *expected_indices; /**< Flat array of size length. */
    float *all_inputs;        /**< Flat 1D array of size length * inputs_length */
} dataset;

/**
 * @brief Allocate and setup a dataset
 *
 * @param dataset_length Number of data of the dataset
 * @param inputs_length Number of input of the data
 *
 * @return newly allocated dataset
 */
dataset *alloc_dataset(size_t dataset_length, size_t inputs_length);

/**
 * @brief Reads a dataset from the specified file.
 *
 * @param filename Path to the file containing the dataset.
 * @return Pointer to the loaded 'dataset' structure, or NULL if an error occurs.
 */
dataset *get_dataset(const char *filename);

/**
 * @brief Creates allocate new dataset and select random copy of data from a source dataset.
 *
 * @param source_dataset Pointer to the source dataset.
 * @param amount Number of data to be created in the new dataset.
 * @return Pointer to the newly created 'dataset' structure.
 */
dataset *get_random_dataset_sample(const dataset *source_dataset, size_t amount);

/**
 * @brief Rotates the data by a specified angle.
 *
 * @param data Float pointer to the start of data to modify.
 * @param width Width of the data (e.g., for image data).
 * @param height Height of the data.
 * @param angle Rotation angle in degrees.
 */
void rotate_data(float *data, int width, int height, float angle);

/**
 * @brief Scales the data by a specified factor.
 *
 * @param data Float pointer to the start of data to modify.
 * @param width Width of the data (e.g., for image data).
 * @param height Height of the data.
 * @param scale Scaling factor.
 */
void scale_data(float *data, int width, int height, float scale);

/**
 * @brief Applies an offset to the data in both x and y directions.
 *
 * @param data Float pointer to the start of data to modify.
 * @param width Width of the data (e.g., for image data).
 * @param height Height of the data.
 * @param offset_x Offset value in the x-direction.
 * @param offset_y Offset value in the y-direction.
 */
void offset_data(float *data, int width, int height, float offset_x, float offset_y);

/**
 * @brief Adds noise to the data with a given intensity and probability.
 *
 * @param data Float pointer to the start of data to modify.
 * @param inputs_length Number of input values in the data element.
 * @param noise_factor Intensity of the noise to be added.
 * @param probability Probability of adding noise to each input value.
 */
void noise_data(float *data, size_t inputs_length, float noise_factor, float probability);

/**
 * @brief Apply activation to a vector.
 *
 * @param a Pointer to the vector.
 * @param b Pointer to the resulting vector.
 * @param length Number of element of the vector.
 * @param activation_function Activation function used to apply activation.
 * @param is_derivative Toggle between derivative calculation and non derivative calculation.
 */
void vector_apply_activation(const float *a, float *b, size_t length, float (*activation_function)(float, bool), bool is_derivative);

/**
 * @brief Compute hadamard product of two vector.
 *
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @param c Pointer to the resulting vector.
 * @param length Number of element of the vector.
 */
void hadamard_product(const float *restrict a, const float *restrict b, float *restrict c, size_t length);

/**
 * @brief Represents a neural network with multiple layers.
 */
typedef struct {
    size_t length;                             /**< Number of layers in the network. */
    size_t inputs_length;                      /**< Number of inputs to the network. */
    size_t total_allocated_memory;               /**< The total memory allocated for the whole neural network. */
    size_t *layer_lengths;                     /**< Number of neuron in each layer. */
    size_t *prev_lengths_sums;                 /**< Number of neuron from all previous layer. */
    size_t *prev_weights_sums;                 /**< Number of weights from all previous layer. */
    float (*activation_function)(float, bool); /**< Pointer to the activation function used in the network. */
    float *delta;                              /**< Error delta for backpropagation. */
    float *weighted_input;                     /**< Weighted input values for the layer. */
    float *output;                             /**< Output values from the layer. */
    float *bias;                               /**< Bias values for the layer. */
    float *weights;                            /**< Weights of the layer in column-major format. */
} neural_network;

/**
 * @brief Allocate and setup a neural_network
 *
 * @param network_length Number of layers in the network.
 * @param layers_length Array specifying the number of neurons in each layer.
 * @param inputs_length Number of inputs to the network.
 *
 * @return Newly allocated data
 */
neural_network *alloc_neural_network(size_t network_length, const size_t *layers_length, size_t inputs_length);

/**
 * @brief Allocates and initializes a new neural network.
 *
 * @param network_length Number of layers in the network.
 * @param layers_length Array specifying the number of neurons in each layer.
 * @param inputs_length Number of inputs to the network.
 * @param activation_function Activation function to be used in the network.
 *
 * @return Pointer to the newly created neural network.
 */
neural_network *get_neural_network(size_t network_length, const size_t *layers_length, size_t inputs_length, float (*activation_function)(float, bool));

/**
 * @brief Allocates and copy a new neural network.
 *
 * @param neural_network The source to copy from
 *
 * @return Pointer to the newly created neural network.
 */
neural_network *copy_neural_network(const neural_network *nn);

/**
 * @brief Computes the output of the neural network for the given inputs.
 *
 * @param nn Pointer to the neural network.
 * @param inputs The inputs to compute.
 *
 * @note The weights and biases are automatically initialized when the network is created using 'get_neural_network'. Ensure the network is created properly before calling this function.
 */
void compute_network(const neural_network *restrict nn, const float *restrict inputs);

/**
 * @brief Applies the softmax function for a specific neuron in the output layer.
 *
 * @param nn Pointer to the neural network.
 * @param neuron_index Index of the neuron to compute the softmax value for.
 * @return The softmax value for the specified neuron.
 */
float softmax(const neural_network *nn, size_t neuron_index);

/**
 * @brief Prints the activation percentages of neurons in the network.
 *
 * @param nn Pointer to the neural network.
 */
void print_activation_percentages(const neural_network *nn);

/**
 * @brief Computes the cost (loss) of the neural network on a test dataset.
 *
 * @param nn Pointer to the neural network.
 * @param test_dataset Pointer to the test dataset.
 * @param num_test Number of test samples to evaluate.
 * @return The computed cost value.
 */
float cost(const neural_network *nn, const dataset *test_dataset, size_t num_test);

/**
 * @brief Performs learning (backpropagation) for a specific layer.
 *
 * @param nn Pointer to the neural network.
 * @param layer_index Index of the layer to perform learning on.
 * @param learn_rate Learning rate for weight updates.
 * @param data Float pointer to the data element used for learning.
 * @param data_expected_index Expected index of the correct output
 *
 * @note The network must be computed using 'compute_network' prior to calling this function.
 */
void layer_learn(const neural_network *nn, size_t layer_index, float learn_rate, const float *data, size_t data_expected_index);

/**
 * @brief Performs backpropagation for a specific layer but add the change in gradient to a array.
 *
 * @param nn Pointer to the neural network.
 * @param layer_weights_gradients Pointer to an array of weights to be added to.
 * @param layer_bias_gradients Pointer to an array of bias to be added to.
 * @param layer_index Index of the layer to perform learning on.
 * @param data Float pointer to the data element used for learning.
 * @param data_expected_index Expected index of the correct output

 *
 * @note The network must be computed using 'compute_network' prior to calling this function.
 */
void layer_learn_collect_gradient(const neural_network *nn, float *restrict layer_weights_gradients, float *restrict layer_bias_gradients, size_t layer_index, const float *data, size_t data_expected_index);

/**
 * @brief Performs stochastic gradient descent to the network.
 *
 * @param nn Pointer to the neural network.
 * @param learn_rate Learning rate for weight updates.
 * @param data Float pointer to the data element used for learning.
 * @param data_expected_index Expected index of the correct output
 *
 * @note The network must be computed using 'compute_network' prior to calling this function.
 */
void stochastic_gd(const neural_network *nn, float learn_rate, const float *data, size_t data_exptected_index);

/**
 * @brief Performs mini-batch gradient descent to the network.
 *
 * @param nn Pointer to the neural network.
 * @param learn_rate Learning rate for weight updates.
 * @param data_batch Pointer to the dataset used for gradient descent.
 *
 * @note The network must be computed using 'compute_network' prior to calling this function.
 */
void mini_batch_gd(neural_network *nn, float learn_rate, const dataset *data_batch);

/**
 * @brief Saves the neural network to a file.
 *
 * @param filename Path to the file where the network will be saved.
 * @param nn Pointer to the neural network to save.
 * @return True if the network was successfully saved, false otherwise.
 */
bool save_network(const char *restrict filename, const neural_network *restrict nn);

/**
 * @brief Loads a neural network from a file.
 *
 * @param filename Path to the file from which the network will be loaded.
 * @param nn Pointer to the neural network structure to load into.
 * @return True if the network was successfully loaded, false otherwise.
 */
bool load_network(const char *restrict filename, const neural_network *restrict nn);

/**
 * @brief Tests the accuracy of the neural network on a test dataset.
 *
 * @param nn Pointer to the neural network.
 * @param test_dataset Pointer to the test dataset.
 * @return The percentage of correct predictions.
 */
float test_network_percent(const neural_network *nn, const dataset *test_dataset);

struct rand_chunk {
    size_t count;
    uint8_t buf[1024];  // NOTE: must be multiple of 256
};

uint8_t randnum_u8(uint8_t range, uint8_t offset);
uint32_t randnum_u32(uint32_t range, uint32_t offset);
float randf(float range, float offset);

extern struct rand_chunk randc;
#endif
