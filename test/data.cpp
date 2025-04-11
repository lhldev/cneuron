#include <gtest/gtest.h>

extern "C" {
#include "cneuron/cneuron.h"
}

TEST(DataTest, CreateData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));

    size_t inputs_length = 10;
    data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    EXPECT_NE(data, nullptr);
    EXPECT_NE(data->inputs, nullptr);
    for (size_t i = 0; i < inputs_length; i++) {
        data->inputs[i] = static_cast<float>(i);
    }
    free_data(data);
    // No crash
}

TEST(DataTest, GetDatasetFileNotFound) {
    dataset_t *dataset = get_dataset("non_existent_file.dat");
    ASSERT_EQ(dataset, nullptr);
}

TEST(DataTest, GetDatasetValidFile) {
    dataset_t *dataset = get_dataset("data/mnist/test/0.dat");
    ASSERT_NE(dataset, nullptr);
    ASSERT_GT(dataset->length, 0);
    ASSERT_GT(dataset->inputs_length, 0);

    ASSERT_NE(dataset, nullptr);
    ASSERT_NE(dataset->datas[0], nullptr);
    ASSERT_NE(dataset->datas[0]->inputs, nullptr);

    free_dataset(dataset);
}

TEST(DataTest, FreeDataset) {
    dataset_t *dataset = get_dataset("data/mnist/test/0.dat");

    free_dataset(dataset);
    // No crash
}

TEST(DataTest, FreeData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));
    data->inputs = (float *)malloc(sizeof(float) * 10);

    free_data(data);
    // No crash
}

TEST(DataTest, CopyData) {
    dataset_t *dataset = get_dataset("data/mnist/test/0.dat");

    data_t *data_copy = get_data_copy(dataset->datas[0], dataset->inputs_length);
    ASSERT_NE(data_copy, nullptr);
    ASSERT_NE(data_copy->inputs, nullptr);

    for (size_t i = 0; i < dataset->inputs_length; i++) {
        ASSERT_FLOAT_EQ(data_copy->inputs[i], dataset->datas[0]->inputs[i]);
    }

    ASSERT_FLOAT_EQ(data_copy->expected_index, dataset->datas[0]->expected_index);

    free_data(data_copy);
    free_dataset(dataset);
}

TEST(DataTest, RotateData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));

    size_t inputs_length = 9;
    data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        data->inputs[i] = static_cast<float>(i) + 1.0f;
    }

    rotate_data(data, 3, 3, 90);
    ASSERT_FLOAT_EQ(data->inputs[0], 7.0f);
    ASSERT_FLOAT_EQ(data->inputs[2], 1.0f);
    ASSERT_FLOAT_EQ(data->inputs[4], 5.0f);

    free_data(data);
}

TEST(DataTest, ScaleData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));

    size_t inputs_length = 9;
    data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        data->inputs[i] = i + 1.0f;
    }

    scale_data(data, 3, 3, 2.0f);
    ASSERT_FLOAT_EQ(data->inputs[0], 5.0f);
    ASSERT_FLOAT_EQ(data->inputs[2], 6.0f);
    ASSERT_FLOAT_EQ(data->inputs[8], 9.0f);

    free_data(data);
}

TEST(DataTest, OffsetData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));

    size_t inputs_length = 9;
    data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        data->inputs[i] = i + 1.0f;
    }

    offset_data(data, 3, 3, 1.0f, 1.0f);
    ASSERT_FLOAT_EQ(data->inputs[0], 0.0f);
    ASSERT_FLOAT_EQ(data->inputs[5], 2.0f);
    ASSERT_FLOAT_EQ(data->inputs[6], 0.0f);
    ASSERT_FLOAT_EQ(data->inputs[8], 5.0f);

    free_data(data);
}

TEST(DataTest, NoiseData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));

    size_t inputs_length = 9;
    data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        data->inputs[i] = i + 1.0f;
    }
    data_t *data_copy = get_data_copy(data, inputs_length);

    bool same = true;
    noise_data(data, inputs_length, 1.0f, 1.0f);
    for (size_t i = 0; i < inputs_length; i++) {
        if (data_copy->inputs[i] != data->inputs[i]) {
            same = false;
            break;
        }
    }
    ASSERT_FALSE(same);

    free_data(data);
    free_data(data_copy);
}

TEST(DataTest, OutputExpected) {
    data_t *data = (data_t *)malloc(sizeof(data_t));
    data->expected_index = 1;

    ASSERT_FLOAT_EQ(output_expected(0, data), 0.0f);
    ASSERT_FLOAT_EQ(output_expected(1, data), 1.0f);

    free(data);
}
