#include <gtest/gtest.h>

extern "C" {
#include "cneuron/cneuron.h"
}

TEST(DataTest, CreateData) {
    data *test_data = (data *)malloc(sizeof(data));

    size_t inputs_length = 10;
    test_data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    EXPECT_NE(test_data, nullptr);
    EXPECT_NE(test_data->inputs, nullptr);
    for (size_t i = 0; i < inputs_length; i++) {
        test_data->inputs[i] = static_cast<float>(i);
    }
    free_data(test_data);
    // No crash
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
    ASSERT_NE(test_dataset->datas[0], nullptr);
    ASSERT_NE(test_dataset->datas[0]->inputs, nullptr);

    free_dataset(test_dataset);
}

TEST(DataTest, FreeDataset) {
    dataset *test_dataset = get_dataset("data/mnist/test/0.dat");

    free_dataset(test_dataset);
    // No crash
}

TEST(DataTest, FreeData) {
    data *test_data = (data *)malloc(sizeof(data));
    test_data->inputs = (float *)malloc(sizeof(float) * 10);

    free_data(test_data);
    // No crash
}

TEST(DataTest, CopyData) {
    dataset *test_dataset = get_dataset("data/mnist/test/0.dat");

    data *data_copy = get_data_copy(test_dataset->datas[0], test_dataset->inputs_length);
    ASSERT_NE(data_copy, nullptr);
    ASSERT_NE(data_copy->inputs, nullptr);

    for (size_t i = 0; i < test_dataset->inputs_length; i++) {
        ASSERT_FLOAT_EQ(data_copy->inputs[i], test_dataset->datas[0]->inputs[i]);
    }

    ASSERT_FLOAT_EQ(data_copy->expected_index, test_dataset->datas[0]->expected_index);

    free_data(data_copy);
    free_dataset(test_dataset);
}

TEST(DataTest, RandomSampleDataset) {
    dataset *test_dataset = get_dataset("data/mnist/test/0.dat");

    dataset *dataset_sample = get_random_dataset_sample(test_dataset, test_dataset->length - 1);
    ASSERT_NE(dataset_sample, nullptr);
    ASSERT_NE(dataset_sample->datas, nullptr);
    ASSERT_NE(dataset_sample->datas[0]->inputs, nullptr);

    free_dataset(dataset_sample);
    free_dataset(test_dataset);
}

TEST(DataTest, RotateData) {
    data *test_data = (data *)malloc(sizeof(data));

    size_t inputs_length = 9;
    test_data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        test_data->inputs[i] = static_cast<float>(i) + 1.0f;
    }

    rotate_data(test_data, 3, 3, 90);
    ASSERT_FLOAT_EQ(test_data->inputs[0], 7.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[2], 1.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[4], 5.0f);

    free_data(test_data);
}

TEST(DataTest, ScaleData) {
    data *test_data = (data *)malloc(sizeof(data));

    size_t inputs_length = 9;
    test_data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        test_data->inputs[i] = i + 1.0f;
    }

    scale_data(test_data, 3, 3, 2.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[0], 5.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[2], 6.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[8], 9.0f);

    free_data(test_data);
}

TEST(DataTest, OffsetData) {
    data *test_data = (data *)malloc(sizeof(data));

    size_t inputs_length = 9;
    test_data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        test_data->inputs[i] = i + 1.0f;
    }

    offset_data(test_data, 3, 3, 1.0f, 1.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[0], 0.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[5], 2.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[6], 0.0f);
    ASSERT_FLOAT_EQ(test_data->inputs[8], 5.0f);

    free_data(test_data);
}

TEST(DataTest, NoiseData) {
    data *test_data = (data *)malloc(sizeof(data));

    size_t inputs_length = 9;
    test_data->inputs = (float *)malloc(sizeof(float) * inputs_length);
    for (size_t i = 0; i < inputs_length; i++) {
        test_data->inputs[i] = i + 1.0f;
    }
    data *data_copy = get_data_copy(test_data, inputs_length);

    bool same = true;
    noise_data(test_data, inputs_length, 1.0f, 1.0f);
    for (size_t i = 0; i < inputs_length; i++) {
        if (data_copy->inputs[i] != test_data->inputs[i]) {
            same = false;
            break;
        }
    }
    ASSERT_FALSE(same);

    free_data(test_data);
    free_data(data_copy);
}

TEST(DataTest, OutputExpected) {
    data *test_data = (data *)malloc(sizeof(data));
    test_data->expected_index = 1;

    ASSERT_FLOAT_EQ(output_expected(0, test_data), 0.0f);
    ASSERT_FLOAT_EQ(output_expected(1, test_data), 1.0f);

    free(test_data);
}
