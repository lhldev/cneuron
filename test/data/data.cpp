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
    unsigned int dataset_length = 0;
    unsigned int inputs_length = 0;
    data_t **dataset = get_dataset("non_existent_file.dat", &dataset_length, &inputs_length);
    EXPECT_EQ(dataset_length, 0);
    EXPECT_EQ(inputs_length, 0);
    ASSERT_EQ(dataset, nullptr);
}

TEST(DataTest, GetDatasetValidFile) {
    unsigned int dataset_length = 0;
    unsigned int inputs_length = 0;
    data_t **dataset = get_dataset("data/mnist/mnist_train.dat", &dataset_length, &inputs_length);
    ASSERT_NE(dataset, nullptr);
    ASSERT_GT(dataset_length, 0);
    ASSERT_GT(inputs_length, 0);

    ASSERT_NE(dataset[0], nullptr);
    ASSERT_NE(dataset[0]->inputs, nullptr);

    free_dataset(dataset, dataset_length);
}

TEST(DataTest, FreeDataset) {
    unsigned int dataset_length = 0;
    unsigned int inputs_length = 0;
    data_t **dataset = get_dataset("data/mnist/mnist_test.dat", &dataset_length, &inputs_length);

    free_dataset(dataset, dataset_length);
    // No crash
}

TEST(DataTest, FreeData) {
    data_t *data = (data_t *)malloc(sizeof(data_t));
    data->inputs = (float *)malloc(sizeof(float) * 10);

    free_data(data);
    // No crash
}


TEST(DataTest, CopyData) {
    unsigned int dataset_length = 0;
    unsigned int inputs_length = 0;
    data_t **dataset = get_dataset("data/mnist/mnist_test.dat", &dataset_length, &inputs_length);

    data_t *data_copy = get_data_copy(dataset[0], inputs_length);
    ASSERT_NE(data_copy, nullptr);
    ASSERT_NE(data_copy->inputs, nullptr);

    for (int i = 0; i < inputs_length; i++) {
        ASSERT_FLOAT_EQ(data_copy->inputs[i], dataset[0]->inputs[i]);
    }

    ASSERT_FLOAT_EQ(data_copy->neuron_index, dataset[0]->neuron_index);

    free_data(data_copy);
    free_dataset(dataset, inputs_length);
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
        if (data_copy->inputs[i] != data->inputs[i]){
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
