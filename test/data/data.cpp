#include <gtest/gtest.h>

extern "C" {
    #include "data/data.h"
}

TEST(DataTest, CreateData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));
    data->inputs = (float *)malloc(sizeof(float) * 10);
    EXPECT_NE(data, nullptr);
    EXPECT_NE(data->inputs, nullptr);
    for (int i = 0; i < 10; i++) {
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
    dataset_t *dataset = get_dataset("data/mnist/mnist_train.dat");
    ASSERT_NE(dataset, nullptr);
    ASSERT_GT(dataset->length, 0);
    ASSERT_GT(dataset->inputs_length, 0);

    ASSERT_NE(dataset, nullptr);
    ASSERT_NE(dataset->datas[0], nullptr);
    ASSERT_NE(dataset->datas[0]->inputs, nullptr);

    free_dataset(dataset);
}

TEST(DataTest, FreeDataset) {
    dataset_t *dataset = get_dataset("data/mnist/mnist_test.dat");

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
    dataset_t *dataset = get_dataset("data/mnist/mnist_test.dat");

    data_t *data_copy = get_data_copy(dataset->datas[0], dataset->inputs_length);
    ASSERT_NE(data_copy, nullptr);
    ASSERT_NE(data_copy->inputs, nullptr);

    for (int i = 0; i < dataset->inputs_length; i++) {
        ASSERT_FLOAT_EQ(data_copy->inputs[i], dataset->datas[0]->inputs[i]);
    }

    ASSERT_FLOAT_EQ(data_copy->neuron_index, dataset->datas[0]->neuron_index);

    free_data(data_copy);
    free_dataset(dataset);
}

TEST(DataTest, RotateData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));
    data->inputs = (float *)malloc(sizeof(float) * 9);
    for (int i = 0; i < 9; i++) {
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
    data->inputs = (float *)malloc(sizeof(float) * 9);
    for (int i = 0; i < 9; i++) {
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
    data->inputs = (float *)malloc(sizeof(float) * 9);
    for (int i = 0; i < 9; i++) {
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
    data->inputs = (float *)malloc(sizeof(float) * 9);
    for (int i = 0; i < 9; i++) {
        data->inputs[i] = i + 1.0f;
    }
    data_t *data_copy  = get_data_copy(data, 9);

    bool same = true;
    noise_data(data, 9, 1.0f, 1.0f);
    for (int i = 0; i < 9; i++) {
        if (data_copy->inputs[i] != data->inputs[i]) {
            same = false;
            break;
        }
    }
    ASSERT_EQ(same, false);

    free_data(data);
    free_data(data_copy);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
