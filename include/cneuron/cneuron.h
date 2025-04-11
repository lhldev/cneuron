#ifndef CNEURON_H
#define CNEURON_H

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Represents a single data element with its inputs and expected output index.
 */
typedef struct {
    float *inputs;         /**< Pointer to an array of input values. */
    size_t expected_index; /**< Index of the expected output label. */
} data_t;

/**
 * @brief Represents a dataset that contains multiple data elements.
 */
typedef struct {
    size_t length;        /**< Number of data elements in the dataset. */
    size_t inputs_length; /**< Number of input values per data element. */
    data_t **datas;       /**< Array of pointers containing the datas */
} dataset_t;

/**
 * @brief Reads a dataset from the specified file.
 *
 * @param filename Path to the file containing the dataset.
 * @return Pointer to the loaded 'dataset_t' structure, or NULL if an error occurs.
 */
dataset_t *get_dataset(const char *filename);

/**
 * @brief Frees all memory associated with a 'dataset_t' structure and its contents.
 *
 * @param dataset Pointer to the dataset to be freed.
 */
void free_dataset(dataset_t *dataset);

/**
 * @brief Frees all memory associated with a 'data_t' structure and its conetents.
 *
 * @param data Pointer to the data element to be freed.
 */
void free_data(data_t *data);

/**
 * @brief Creates a copy of a 'data_t' structure.
 *
 * @param data Pointer to the original data element to copy.
 * @param inputs_length Number of input values in the data element.
 * @return Pointer to the newly created copy of the 'data_t' structure.
 */
data_t *get_data_copy(const data_t *data, size_t inputs_length);

/**
 * @brief Rotates the data by a specified angle.
 *
 * @param data Pointer to the data element to modify.
 * @param width Width of the data (e.g., for image data).
 * @param height Height of the data.
 * @param angle Rotation angle in degrees.
 */
void rotate_data(data_t *data, int width, int height, float angle);

/**
 * @brief Scales the data by a specified factor.
 *
 * @param data Pointer to the data element to modify.
 * @param width Width of the data (e.g., for image data).
 * @param height Height of the data.
 * @param scale Scaling factor.
 */
void scale_data(data_t *data, int width, int height, float scale);

/**
 * @brief Applies an offset to the data in both x and y directions.
 *
 * @param data Pointer to the data element to modify.
 * @param width Width of the data (e.g., for image data).
 * @param height Height of the data.
 * @param offset_x Offset value in the x-direction.
 * @param offset_y Offset value in the y-direction.
 */
void offset_data(data_t *data, int width, int height, float offset_x, float offset_y);

/**
 * @brief Adds noise to the data with a given intensity and probability.
 *
 * @param data Pointer to the data element to modify.
 * @param inputs_length Number of input values in the data element.
 * @param noise_factor Intensity of the noise to be added.
 * @param probability Probability of adding noise to each input value.
 */
void noise_data(data_t *data, size_t inputs_length, float noise_factor, float probability);

/**
 * @brief Computes the expected output value for the data element.
 *
 * @param index Index of the output neuron to check.
 * @param data Pointer to the data element.
 * @return Floating-point value representing the expected output.
 */
float output_expected(size_t index, const data_t *data);

/**
 * @brief Represents a single layer in a neural network.
 */
typedef struct layer {
    float *delta;             /**< Error delta for backpropagation. */
    float *weighted_input;    /**< Weighted input values for the layer. */
    float *weights;           /**< Weights of the layer in column-major format. */
    float *bias;              /**< Bias values for the layer. */
    float *output;            /**< Output values from the layer. */
    struct layer *prev_layer; /**< Pointer to the previous layer in the network. */
    struct layer *next_layer; /**< Pointer to the next layer in the network. */
    size_t length;            /**< Number of neurons in this layer. */
} layer_t;

/**
 * @brief Represents a neural network with multiple layers.
 */
typedef struct {
    layer_t **layers;                          /**< Array of pointers to layers in the network. */
    size_t length;                             /**< Number of layers in the network. */
    size_t inputs_length;                      /**< Number of inputs to the network. */
    float (*activation_function)(float, bool); /**< Pointer to the activation function used in the network. */
} neural_network_t;

/**
 * @brief Generates a random floating-point number within a given range.
 *
 * @param min Minimum value for the random number.
 * @param max Maximum value for the random number.
 * @return A random float between min and max.
 */
float random_float(float min, float max);

/**
 * @brief Multiplies two matrices stored in column-major format.
 *
 * @param a Pointer to the first matrix.
 * @param b Pointer to the second matrix.
 * @param c Pointer to the resulting matrix.
 * @param rows_a Number of rows in matrix 'a'.
 * @param cols_a Number of columns in matrix 'a' (also rows in matrix 'b').
 * @param cols_b Number of columns in matrix 'b'.
 */
void matrix_multiply(const float *a, const float *b, float *c, size_t rows_a, size_t cols_a, size_t cols_b);

/**
 * @brief Allocates and initializes a new layer.
 *
 * @param length Number of neurons in this layer.
 * @param prev_length Number of neurons in the previous layer.
 * @return Pointer to the newly created layer.
 */
layer_t *get_layer(size_t length, size_t prev_length);

/**
 * @brief Allocates and initializes a new neural network.
 *
 * @param layer_length Number of layers in the network.
 * @param layer_lengths Array specifying the number of neurons in each layer.
 * @param inputs_length Number of inputs to the network.
 * @param activation_function Activation function to be used in the network.
 * @return Pointer to the newly created neural network.
 */
neural_network_t *get_neural_network(size_t layer_length, const size_t *layer_lengths, size_t inputs_length, float (*activation_function)(float, bool));

/**
 * @brief Frees all memory associated with a 'layer_t' structure and its conetents.
 *
 * @param layer Pointer to the layer to be freed.
 */
void free_layer(layer_t *layer);

/**
 * @brief Frees all memory associated with a 'neural_network_t' structure and its conetents.
 *
 * @param nn Pointer to the neural network to be freed.
 */
void free_neural_network(neural_network_t *nn);

/**
 * @brief Computes the output of the neural network for the given inputs.
 *
 * @param nn Pointer to the neural network.
 * @param inputs Array of input values.
 *
 * @note The weights and biases are automatically initialized when the network is created using 'get_neural_network'. Ensure the network is created properly before calling this function.
 */
void compute_network(neural_network_t *nn, const float *inputs);

/**
 * @brief Applies the softmax function for a specific neuron in the output layer.
 *
 * @param nn Pointer to the neural network.
 * @param neuron_index Index of the neuron to compute the softmax value for.
 * @return The softmax value for the specified neuron.
 */
float softmax(neural_network_t *nn, size_t neuron_index);

/**
 * @brief Prints the activation percentages of neurons in the network.
 *
 * @param nn Pointer to the neural network.
 */
void print_activation_percentages(neural_network_t *nn);

/**
 * @brief Computes the cost (loss) of the neural network on a test dataset.
 *
 * @param nn Pointer to the neural network.
 * @param test_dataset Pointer to the test dataset.
 * @param num_test Number of test samples to evaluate.
 * @return The computed cost value.
 */
float cost(neural_network_t *nn, const dataset_t *test_dataset, size_t num_test);

/**
 * @brief Performs learning (backpropagation) for a specific layer.
 *
 * @param nn Pointer to the neural network.
 * @param layer_index Index of the layer to perform learning on.
 * @param learn_rate Learning rate for weight updates.
 * @param data Pointer to the data element used for learning.
 * @param activation_function Pointer to the activation function.
 *
 * @note The network must be computed using 'compute_network' prior to calling this function.
 */
void layer_learn(neural_network_t *nn, size_t layer_index, float learn_rate, const data_t *data, float (*activation_function)(float, bool));

/**
 * @brief Performs learning (backpropagation) for the entire network.
 *
 * @param nn Pointer to the neural network.
 * @param learn_rate Learning rate for weight updates.
 * @param data Pointer to the data element used for learning.
 *
 * @note The network must be computed using 'compute_network' prior to calling this function.
 */
void learn(neural_network_t *nn, float learn_rate, const data_t *data);

/**
 * @brief Saves the neural network to a file.
 *
 * @param filename Path to the file where the network will be saved.
 * @param nn Pointer to the neural network to save.
 * @return True if the network was successfully saved, false otherwise.
 */
bool save_network(const char *filename, neural_network_t *nn);

/**
 * @brief Loads a neural network from a file.
 *
 * @param filename Path to the file from which the network will be loaded.
 * @param nn Pointer to the neural network structure to load into.
 * @return True if the network was successfully loaded, false otherwise.
 */
bool load_network(const char *filename, neural_network_t *nn);

/**
 * @brief Tests the accuracy of the neural network on a test dataset.
 *
 * @param nn Pointer to the neural network.
 * @param test_dataset Pointer to the test dataset.
 * @return The percentage of correct predictions.
 */
float test_network_percent(neural_network_t *nn, const dataset_t *test_dataset);

#endif
