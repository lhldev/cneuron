#include <gtest/gtest.h>

extern "C" {
#include "cneuron/cneuron.h"
}

TEST(DataTest, GetDatasetFileNotFound) {
    dataset *test_dataset = get_dataset("non_existent_file.dat");
    ASSERT_EQ(test_dataset, nullptr);
}

TEST(DataTest, GetDatasetValidFile) {
    dataset *test_dataset = get_dataset("data/mnist/test/0.dat");
    ASSERT_NE(test_dataset, nullptr);
    ASSERT_GT(test_dataset->length, 0);
    ASSERT_GT(test_dataset->inputs_length, 0);

    ASSERT_NE(test_dataset, nullptr);
    ASSERT_NE(test_dataset->all_inputs, nullptr);

    free(test_dataset);
}

TEST(DataTest, FreeDataset) {
    dataset *test_dataset = get_dataset("data/mnist/test/0.dat");

    free(test_dataset);
    // No crash
}

TEST(DataTest, RandomSampleDataset) {
    dataset *test_dataset = get_dataset("data/mnist/test/0.dat");

    dataset *dataset_sample = get_random_dataset_sample(test_dataset, test_dataset->length - 1);
    ASSERT_NE(dataset_sample, nullptr);
    ASSERT_NE(dataset_sample->all_inputs, nullptr);

    free(dataset_sample);
    free(test_dataset);
}

TEST(DataTest, RotateData) {
    size_t inputs_length = 9;
    float *test_data = (float *)malloc(sizeof(float) * inputs_length);

    for (size_t i = 0; i < inputs_length; i++) {
        test_data[i] = static_cast<float>(i) + 1.0f;
    }

    rotate_data(test_data, 3, 3, 90);
    ASSERT_FLOAT_EQ(test_data[0], 7.0f);
    ASSERT_FLOAT_EQ(test_data[2], 1.0f);
    ASSERT_FLOAT_EQ(test_data[4], 5.0f);

    free(test_data);
}

TEST(DataTest, ScaleData) {
    size_t inputs_length = 9;
    float *test_data = (float *)malloc(sizeof(float) * inputs_length);

    for (size_t i = 0; i < inputs_length; i++) {
        test_data[i] = i + 1.0f;
    }

    scale_data(test_data, 3, 3, 2.0f);
    ASSERT_FLOAT_EQ(test_data[0], 5.0f);
    ASSERT_FLOAT_EQ(test_data[2], 6.0f);
    ASSERT_FLOAT_EQ(test_data[8], 9.0f);

    free(test_data);
}

TEST(DataTest, OffsetData) {
    size_t inputs_length = 9;
    float *test_data = (float *)malloc(sizeof(float) * inputs_length);

    for (size_t i = 0; i < inputs_length; i++) {
        test_data[i] = i + 1.0f;
    }

    offset_data(test_data, 3, 3, 1.0f, 1.0f);
    ASSERT_FLOAT_EQ(test_data[0], 0.0f);
    ASSERT_FLOAT_EQ(test_data[5], 2.0f);
    ASSERT_FLOAT_EQ(test_data[6], 0.0f);
    ASSERT_FLOAT_EQ(test_data[8], 5.0f);

    free(test_data);
}

TEST(DataTest, NoiseData) {
    size_t inputs_length = 9;
    float *test_data = (float *)malloc(sizeof(float) * inputs_length);
    float *data_copy = (float *)malloc(sizeof(float) * inputs_length);

    for (size_t i = 0; i < inputs_length; i++) {
        test_data[i] = i + 1.0f;
        data_copy[i] = i + 1.0f;
    }

    bool same = true;
    noise_data(test_data, inputs_length, 1.0f, 1.0f);
    for (size_t i = 0; i < inputs_length; i++) {
        if (data_copy[i] != test_data[i]) {
            same = false;
            break;
        }
    }
    ASSERT_FALSE(same);

    free(test_data);
    free(data_copy);
}
