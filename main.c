#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"
#include <dirent.h>
#include <sys/types.h>
#include <time.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define IMAGE_SIZE 28
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct
{
    double delta; // for backpropagation
    double weightedInput;
    double *weights;
    int numWeights;
    double bias;
    double output;
} Neuron;

typedef struct
{
    int size;
    Neuron *neurons;
} Layer;

typedef struct
{
    Layer inputLayer;
    Layer *hiddenLayers;
    Layer outputLayer;
    int numHiddenLayer;
    double (*activationFunction)(double, int);
} NeuralNetwork;

typedef struct
{
    double inputs[IMAGE_SIZE * IMAGE_SIZE];
    int expected;
} Data;

double sigmoid(double val, int isDeravative)
{
    double result = 1 / (1 + exp(-val));
    if (isDeravative == 1)
    {
        return result * (1 - result);
    }
    return result;
}

double ReLU(double val, int isDeravative)
{
    if (isDeravative)
    {
        return (val > 0) ? 1 : 0;
    }
    return max(0, val);
}

double calcOutput(Layer *previousLayer, Neuron *neuron, double (*activationFunction)(double, int))
{
    neuron->output = 0.0;
    neuron->weightedInput = 0.0;
    for (int i = 0; i < previousLayer->size; i++)
    {
        neuron->weightedInput += previousLayer->neurons[i].output * neuron->weights[i];
    }
    neuron->weightedInput += neuron->bias;
    neuron->output = activationFunction(neuron->weightedInput, 0);
    return neuron->output;
}

void calcOutputLayer(Layer *previousLayer, Layer *currentLayer, double (*activationFunction)(double, int))
{
    for (int i = 0; i < currentLayer->size; i++)
    {
        calcOutput(previousLayer, &currentLayer->neurons[i], activationFunction);
    }
}

void initialiseLayer(Layer *layer, int inputSize)
{
    for (int i = 0; i < layer->size; i++)
    {
        layer->neurons[i].weights = malloc(sizeof(double) * inputSize);
        layer->neurons[i].numWeights = inputSize;
        for (int j = 0; j < inputSize; j++)
        {
            layer->neurons[i].weights[j] = ((double)rand() / RAND_MAX * 2 - 1);
        }
        layer->neurons[i].delta = 0.0;
        layer->neurons[i].bias = 0.0;
        layer->neurons[i].output = 0.0;
        layer->neurons[i].weightedInput = 0.0;
    }
}

void initialiseNeuralNetwork(NeuralNetwork *nn, int numHiddenLayer, int *hiddenLayerSizes, int outputLayerSize, int numInput, double (*activationFunction)(double, int))
{
    nn->inputLayer.neurons = malloc(sizeof(Neuron) * numInput);
    nn->inputLayer.size = numInput;
    nn->hiddenLayers = malloc(sizeof(Layer) * numHiddenLayer);
    nn->numHiddenLayer = numHiddenLayer;

    for (int i = 0; i < numHiddenLayer; i++)
    {
        nn->hiddenLayers[i].neurons = malloc(sizeof(Neuron) * hiddenLayerSizes[i]);
        nn->hiddenLayers[i].size = hiddenLayerSizes[i];
    }

    nn->outputLayer.neurons = malloc(sizeof(Neuron) * outputLayerSize);
    nn->outputLayer.size = outputLayerSize;

    for (int i = 0; i < numHiddenLayer; i++)
    {
        initialiseLayer(&nn->hiddenLayers[i], (i == 0) ? numInput : hiddenLayerSizes[i - 1]);
    }
    initialiseLayer(&nn->inputLayer, numInput);
    initialiseLayer(&nn->outputLayer, (numHiddenLayer == 0) ? numInput : hiddenLayerSizes[numHiddenLayer - 1]);
    nn->activationFunction = activationFunction;
}

void freeLayer(Layer *layer)
{
    for (int i = 0; i < layer->size; i++)
    {
        free(layer->neurons[i].weights);
    }
    free(layer->neurons);
}

void freeNeuralNetwork(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->numHiddenLayer; i++)
    {
        freeLayer(&nn->hiddenLayers[i]);
    }
    freeLayer(&nn->outputLayer);
    freeLayer(&nn->inputLayer);
    free(nn->hiddenLayers);
}

void addInputs(NeuralNetwork *nn, double *inputs)
{
    for (int i = 0; i < nn->inputLayer.size; i++)
    {
        nn->inputLayer.neurons[i].output = inputs[i];
    }
}

void computeNetwork(NeuralNetwork *nn)
{
    if (nn->numHiddenLayer == 0)
    {
        calcOutputLayer(&nn->inputLayer, &nn->outputLayer, nn->activationFunction);
    }
    else
    {
        Layer *currLayer = &nn->inputLayer;
        for (int i = 0; i < nn->numHiddenLayer; i++)
        {
            calcOutputLayer(currLayer, &nn->hiddenLayers[i], nn->activationFunction);
            currLayer = &nn->hiddenLayers[i];
        }
        calcOutputLayer(currLayer, &nn->outputLayer, nn->activationFunction);
    }
}

double outputNeuronPercentActivate(NeuralNetwork *nn, int neuronIndex)
{
    double sumActivation = 0.0;
    for (int i = 0; i < nn->outputLayer.size; i++)
    {
        sumActivation += nn->outputLayer.neurons[i].output;
    }
    return nn->outputLayer.neurons[neuronIndex].output / sumActivation * 100;
}

void printOutputNeuronsPercentActivate(NeuralNetwork *nn)
{

    double *percentages = malloc(nn->outputLayer.size * sizeof(double));
    int *indices = malloc(nn->outputLayer.size * sizeof(int));
    if (percentages == NULL || indices == NULL)
    {
        printf("Memory allocation failed\n");
        return;
    }

    // Store the activation percentages and indices
    for (int i = 0; i < nn->outputLayer.size; i++)
    {
        percentages[i] = outputNeuronPercentActivate(nn, i);
        indices[i] = i;
    }

    // Selection sort for percentages and corresponding indices
    for (int i = 0; i < nn->outputLayer.size - 1; i++)
    {
        int max_idx = i;
        for (int j = i + 1; j < nn->outputLayer.size; j++)
        {
            if (percentages[j] > percentages[max_idx])
            {
                max_idx = j;
            }
        }
        // Swap percentages
        double temp = percentages[max_idx];
        percentages[max_idx] = percentages[i];
        percentages[i] = temp;
        // Swap indices
        int temp_idx = indices[max_idx];
        indices[max_idx] = indices[i];
        indices[i] = temp_idx;
    }

    // Print the sorted percentages with neuron indices
    for (int i = 0; i < nn->outputLayer.size; i++)
    {
        printf(" (%d = %.2f%%) ", indices[i], percentages[i]);
    }

    printf("\n");

    free(percentages);
    free(indices);
}

double outputNeuronExpected(int neuronIndex, Data *data)
{
    if (data->expected == neuronIndex)
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}

double cost(NeuralNetwork *nn, Data *trainingData, int numData)
{
    double cost = 0;

    for (int i = 0; i < numData; i++)
    {
        addInputs(nn, trainingData[i].inputs);
        computeNetwork(nn);
        for (int j = 0; j < nn->outputLayer.size; j++)
        {
            double output = nn->outputLayer.neurons[j].output;
            cost += (output - outputNeuronExpected(j, &trainingData[i])) * (output - outputNeuronExpected(j, &trainingData[i]));
        }
    }
    return cost / numData;
}

void printResult(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->outputLayer.size; i++)
    {
        printf("%f ", nn->outputLayer.neurons[i].output);
    }
}

void layerLearnOutput(NeuralNetwork *nn, Layer *previousLayer, Layer *layer, double learnRate, Data *trainingData, double (*activationFunction)(double, int))
{
    addInputs(nn, trainingData->inputs);
    computeNetwork(nn);
    for (int i = 0; i < layer->size; i++)
    {
        double neuronOutput = layer->neurons[i].output;
        double targetOutput = outputNeuronExpected(i, trainingData);

        layer->neurons[i].delta = 2 * (neuronOutput - targetOutput) * activationFunction(layer->neurons[i].weightedInput, 1);

        for (int j = 0; j < layer->neurons[i].numWeights; j++)
        {
            double input = previousLayer->neurons[j].output;
            layer->neurons[i].weights[j] -= layer->neurons[i].delta * input * learnRate;
        }

        layer->neurons[i].bias -= layer->neurons[i].delta * learnRate;
    }
}

void layerLearnIntermediate(NeuralNetwork *nn, Layer *previousLayer, Layer *layer, Layer *nextLayer, double learnRate, double (*activationFunction)(double, int))
{
    for (int i = 0; i < layer->size; i++)
    {
        layer->neurons[i].delta = 0.0;
        for (int j = 0; j < nextLayer->size; j++)
        {
            double weightOfNextNeuron = nextLayer->neurons[j].weights[i];
            double deltaOfNextNeuron = nextLayer->neurons[j].delta;
            layer->neurons[i].delta += weightOfNextNeuron * deltaOfNextNeuron * activationFunction(layer->neurons[i].weightedInput, 1);
        }

        for (int k = 0; k < layer->neurons[i].numWeights; k++)
        {
            double input = previousLayer->neurons[k].output;
            layer->neurons[i].weights[k] -= layer->neurons[i].delta * input * learnRate;
        }

        layer->neurons[i].bias -= layer->neurons[i].delta * learnRate;
    }
}

void learn(NeuralNetwork *nn, double learnRate, Data *trainingData, int numData)
{
    for (int i = 0; i < numData; i++)
    {
        layerLearnOutput(nn, (nn->numHiddenLayer == 0) ? &nn->inputLayer : &nn->hiddenLayers[nn->numHiddenLayer - 1], &nn->outputLayer, learnRate, &trainingData[i], nn->activationFunction);
        for (int j = nn->numHiddenLayer - 1; j >= 0; j--)
        {
            layerLearnIntermediate(nn, (j == 0) ? &nn->inputLayer : &nn->hiddenLayers[j - 1], &nn->hiddenLayers[j], (j == nn->numHiddenLayer - 1) ? &nn->outputLayer : &nn->hiddenLayers[j + 1], learnRate, nn->activationFunction);
        }
    }
}

unsigned char *rotateImage(unsigned char *image, int width, int height, float angle)
{
    int newWidth = width;
    int newHeight = height;
    float rad = angle * M_PI / 180.0f;
    float cosAngle = cos(rad);
    float sinAngle = sin(rad);
    unsigned char *newImage = (unsigned char *)malloc(newWidth * newHeight);

    for (int y = 0; y < newHeight; y++)
    {
        for (int x = 0; x < newWidth; x++)
        {
            int centerX = newWidth / 2;
            int centerY = newHeight / 2;
            int srcX = (int)((x - centerX) * cosAngle - (y - centerY) * sinAngle + centerX);
            int srcY = (int)((x - centerX) * sinAngle + (y - centerY) * cosAngle + centerY);

            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height)
            {
                newImage[y * newWidth + x] = image[srcY * width + srcX];
            }
            else
            {
                newImage[y * newWidth + x] = 0; // Set background color to black
            }
        }
    }
    return newImage;
}

unsigned char *scaleImage(unsigned char *image, int width, int height, float scale)
{
    int scaleWidth = width * scale;
    int scaleHeight = height * scale;
    unsigned char *scaleImage = (unsigned char *)malloc(scaleWidth * scaleHeight);
    unsigned char *newImage = (unsigned char *)malloc(width * height);

    for (int y = 0; y < scaleHeight; y++)
    {
        for (int x = 0; x < scaleWidth; x++)
        {
            int srcX = (int)(x / scale);
            int srcY = (int)(y / scale);

            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height)
            {
                scaleImage[y * scaleWidth + x] = image[srcY * width + srcX];
            }
            else
            {
                scaleImage[y * scaleWidth + x] = 0; // Set background color to black
            }
        }
    }
    int offSetX = (scaleWidth - width) / 2;
    int offSetY = (scaleHeight - height) / 2;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int scaleX = x + offSetX;
            int scaleY = y + offSetY;
            if (scaleX >= 0 && scaleX < scaleWidth && scaleY >= 0 && scaleY < scaleHeight)
            {
                newImage[y * width + x] = scaleImage[scaleY * scaleWidth + scaleX];
            }
            else
            {
                newImage[y * width + x] = 0;
            }
        }
    }

    free(scaleImage);
    return newImage;
}

unsigned char *addOffset(unsigned char *image, int width, int height, int offsetX, int offsetY)
{
    unsigned char *newImage = (unsigned char *)malloc(width * height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int newX = x + offsetX;
            int newY = y + offsetY;

            if (newX >= 0 && newX < width && newY >= 0 && newY < height)
            {
                newImage[y * width + x] = image[newY * width + newX];
            }
            else
            {
                newImage[y * width + x] = 0; // Set background color to black
            }
        }
    }
    return newImage;
}

unsigned char *addNoise(unsigned char *image, int width, int height, float noiseFactor, float probability)
{
    unsigned char *newImage = (unsigned char *)malloc(width * height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            float randomValue = (float)rand() / RAND_MAX; // Generate a random value between 0 and 1
            if (randomValue <= probability)
            {
                int noise = (int)(rand() % 256 * noiseFactor);
                int newValue = image[y * width + x] + noise;

                if (newValue < 0)
                    newValue = 0;
                if (newValue > 255)
                    newValue = 255;

                newImage[y * width + x] = newValue;
            }
            else
            {
                newImage[y * width + x] = image[y * width + x];
            }
        }
    }
    return newImage;
}

Data dataFromImage(char *path, float angle, float scale, int offSetX, int offSetY, float noiseFactor, float probability)
{
    Data data;

    int width, height, channels;
    unsigned char *image = stbi_load(path, &width, &height, &channels, STBI_grey);
    if (image == NULL)
    {
        fprintf(stderr, "Failed to load image: %s\n", path);
        return data;
    }

    if (width != IMAGE_SIZE || height != IMAGE_SIZE || channels != 1)
    {
        fprintf(stderr, "Invalid image dimensions or channels: %s\n", path);
        stbi_image_free(image);
        return data;
    }

    unsigned char *scaledImage = scaleImage(image, width, height, scale);
    unsigned char *rotatedImage = rotateImage(scaledImage, width, height, angle);
    unsigned char *offsetImage = addOffset(rotatedImage, width, height, offSetX, offSetY);
    unsigned char *noisyImage = addNoise(offsetImage, width, height, noiseFactor, probability);

    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++)
    {
        data.inputs[i] = (double)image[i] / 255.0;
    }

    char *filename = strrchr(path, '/');
    if (filename == NULL)
    {
        filename = path;
    }
    else
    {
        filename--;
    }
    data.expected = atoi(filename);

    stbi_image_free(image);
    free(scaledImage);
    free(rotatedImage);
    free(offsetImage);
    free(noisyImage);
    return data;
}
float randomFloat(float min, float max)
{
    return (float)rand() / RAND_MAX * (max - min) + min;
}
Data *populateDataSet(int *numData, int maxEachDigit, int *currentPos)
{
    *numData = 0;
    Data *dataSet = malloc(sizeof(Data));
    int oldCurrent = *currentPos;
    struct dirent *entry;
    int count = 0;
    for (int i = 0; i < 10; i++)
    {
        char subdirectory[30];
        sprintf(subdirectory, "data/train/%d", i);
        DIR *dir = opendir(subdirectory);
        if (dir == NULL)
        {
            fprintf(stderr, "Failed to open directory: %s\n", subdirectory);
            exit(1);
        }
        count = 0;
        while ((entry = readdir(dir)) != NULL && count - oldCurrent < maxEachDigit)
        {
            if (count < oldCurrent)
            {
                count++;
                continue;
            }
            if (entry->d_type == DT_REG)
            {
                count++;
                char filepath[256];
                sprintf(filepath, "%s/%s", subdirectory, entry->d_name);
                Data newData = dataFromImage(filepath, randomFloat(-25, 25), randomFloat(0.8, 1.2), randomFloat(-10, 10), randomFloat(-10, 10), randomFloat(0, 0.2), randomFloat(0, 0.2));
                *numData += 1;
                dataSet = realloc(dataSet, sizeof(Data) * (*numData));
                dataSet[*numData - 1] = newData;
            }
        }
        closedir(dir);
    }
    *currentPos += count - oldCurrent;
    if (entry == NULL)
    {
        // no more files in this directory
        *currentPos = 0;
    }
    return dataSet;
}

void saveNetwork(const char *filename, NeuralNetwork *network)
{
    FILE *file = fopen(filename, "wb");
    if (file == NULL)
    {
        printf("Error opening file for writing\n");
        return;
    }

    fwrite(&(network->inputLayer.size), sizeof(int), 1, file);
    fwrite(&(network->numHiddenLayer), sizeof(int), 1, file);

    for (int i = 0; i < network->numHiddenLayer; i++)
    {
        fwrite(&(network->hiddenLayers[i].size), sizeof(int), 1, file);
        for (int j = 0; j < network->hiddenLayers[i].size; j++)
        {
            fwrite(network->hiddenLayers[i].neurons[j].weights, sizeof(double), network->hiddenLayers[i].neurons[j].numWeights, file);
            fwrite(&(network->hiddenLayers[i].neurons[j].bias), sizeof(double), 1, file);
        }
    }

    // Output layer
    fwrite(&(network->outputLayer.size), sizeof(int), 1, file);
    for (int i = 0; i < network->outputLayer.size; i++)
    {
        fwrite(network->outputLayer.neurons[i].weights, sizeof(double), network->outputLayer.neurons[i].numWeights, file);
        fwrite(&(network->outputLayer.neurons[i].bias), sizeof(double), 1, file);
    }

    fclose(file);
}

void loadNetwork(const char *filename, NeuralNetwork *network)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Error opening file for reading\n");
        return;
    }

    int checkVal = 0;
    fread(&checkVal, sizeof(int), 1, file);
    if (checkVal != network->inputLayer.size)
    {
        printf("Number of input layer not compatiable with save file, expected: %d\n", checkVal);
        return;
    }
    fread(&checkVal, sizeof(int), 1, file);

    if (checkVal != network->numHiddenLayer)
    {
        printf("Number of hidden layer not compatable with save file, expected: %d\n", checkVal);
        return;
    }
    for (int i = 0; i < network->numHiddenLayer; i++)
    {
        checkVal = 0;
        fread(&checkVal, sizeof(int), 1, file);
        if (checkVal != network->hiddenLayers[i].size)
        {
            printf("Number of hidden layer neuron not compatable with save file, expected: %d\n", checkVal);
            return;
        }
        for (int j = 0; j < network->hiddenLayers[i].size; j++)
        {
            fread(network->hiddenLayers[i].neurons[j].weights, sizeof(double), network->hiddenLayers[i].neurons[j].numWeights, file);
            fread(&(network->hiddenLayers[i].neurons[j].bias), sizeof(double), 1, file);
        }
    }

    // Output layer
    checkVal = 0;
    fread(&checkVal, sizeof(int), 1, file);
    if (checkVal != network->outputLayer.size)
    {
        printf("Number of output layer neuron not compatable with save file, expected: %d\n", checkVal);
        return;
    }
    for (int i = 0; i < network->outputLayer.size; i++)
    {
        fread(network->outputLayer.neurons[i].weights, sizeof(double), network->outputLayer.neurons[i].numWeights, file);
        fread(&(network->outputLayer.neurons[i].bias), sizeof(double), 1, file);
    }
    fclose(file);
}

double testNetworkPercent(NeuralNetwork *nn)
{
    int tested = 0;
    int correct = 0;
    for (int i = 0; i < 10; i++)
    {
        char subdirectory[30];
        sprintf(subdirectory, "data/test/%d", i);
        DIR *dir = opendir(subdirectory);
        struct dirent *entry;
        if (dir == NULL)
        {
            fprintf(stderr, "Failed to open directory: %s\n", subdirectory);
            exit(1);
        }
        while ((entry = readdir(dir)) != NULL)
        {
            if (entry->d_type == DT_REG)
            {
                tested++;
                char filepath[256];
                sprintf(filepath, "%s/%s", subdirectory, entry->d_name);
                Data testData = dataFromImage(filepath, 0, 1, 0, 0, 0, 0);
                addInputs(nn, testData.inputs);
                computeNetwork(nn);
                if (outputNeuronPercentActivate(nn, testData.expected) >= 50.0)
                {
                    correct++;
                }
            }
        }
        closedir(dir);
    }
    return (double)correct * 100.0 / (double)tested;
}

void train(NeuralNetwork *network, double learnRate, int *numData, int maxEach, int learnAmount, int epochAmount)
{
    int currentPos = 0;
    Data *trainingData = populateDataSet(numData, maxEach, &currentPos);
    clock_t startTime = clock();
    for (int i = 0; i <= learnAmount; i++)
    {
        if (i % epochAmount == 0 && i != 0)
        {
            double newCost = cost(network, trainingData, *numData);
            clock_t elapsedMs = clock() - startTime;
            double elapsedS = (double)elapsedMs / CLOCKS_PER_SEC;
            double speed = (double)*numData / elapsedS * (double)epochAmount;
            printf("Epoch learned %d, cost: %f, elapsed time: %.2fs, speed: %.2f Data/s \n", i, newCost, elapsedS, speed);
            startTime = clock();
            free(trainingData);
            trainingData = populateDataSet(numData, maxEach, &currentPos);
        }
        learn(network, learnRate, trainingData, *numData);
    }
    free(trainingData);
}

// TODO:
// gpu parallelization

int main()
{
    srand(time(NULL));
    int numInput = IMAGE_SIZE * IMAGE_SIZE;
    int numHiddenLayer = 2;
    int *hiddenLayerSizes = malloc(numHiddenLayer * sizeof(int));
    hiddenLayerSizes[0] = 100;
    hiddenLayerSizes[1] = 16;
    int outputLayerSize = 10;
    double (*activationFunction)(double, int) = &sigmoid;

    NeuralNetwork network;
    initialiseNeuralNetwork(&network, numHiddenLayer, hiddenLayerSizes, outputLayerSize, numInput, activationFunction);

    int numData = 0;

    // Parameters
    int maxEach = 10;
    double learnRate = 0.03;
    int learnAmount = 10000;
    int epochAmount = 60;

    char cmd[100];
    FILE *fp;
    double userInput[IMAGE_SIZE * IMAGE_SIZE];
    while (1)
    {
        printf("cmd: ");
        if (scanf("%99s", cmd) != 1)
        {
            printf("Invalid input format. Please try again.\n");
            continue;
        }
        if (cmd[0] == 'q')
        {
            break;
        }
        else if (cmd[0] == 's')
        {
            saveNetwork("output/nn.dat", &network);
            printf("Neural network saved!\n");
        }
        else if (cmd[0] == 'l')
        {
            loadNetwork("output/nn.dat", &network);
            printf("Neural network loaded!\n");
        }
        else if (cmd[0] == 't')
        {
            train(&network, learnRate, &numData, maxEach, learnAmount, epochAmount);
            printf("Training completed. Trained for %d times.\n", learnAmount);
        }
        else if (cmd[0] == 'T')
        {
            printf("Testing neural network...\n");
            printf("Network is %.2f%% correct!\n", testNetworkPercent(&network));
        }
        else if (cmd[0] == 'i')
        {
            printf("Enter your input in the window and press enter...\n");
            while (1)
            {
                char *file_contents;
                system("python input.py");

                // Open the file for reading
                fp = fopen("output/grid_array.txt", "r");
                if (fp == NULL)
                {
                    printf("Error opening file\n");
                    fclose(fp);
                    break;
                }

                char quitFlag;
                fscanf(fp, "%s", &quitFlag);
                if (quitFlag == 'q')
                {
                    fclose(fp);
                    break;
                }

                double genericDouble = 0.0;
                int count = 0;
                while (fscanf(fp, "%lf", &genericDouble) == 1)
                {
                    if (count > IMAGE_SIZE * IMAGE_SIZE)
                    {
                        printf("Warning parsing input\n");
                        break;
                    }
                    userInput[count++] = genericDouble;
                }
                fclose(fp);

                addInputs(&network, userInput);
                computeNetwork(&network);
                printOutputNeuronsPercentActivate(&network);
            }
        }
        else
        {
            printf("Command not recognised. \nt - train \nT - test network\ni - insert input \nl - load \ns - save \nq - quit\n");
            continue;
        }
    }
    freeNeuralNetwork(&network);
    if (numHiddenLayer > 0)
    {
        free(hiddenLayerSizes);
    }
    return 0;
}
