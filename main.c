#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#include <dirent.h>
#include <sys/types.h>

#define max(a, b) ((a) > (b) ? (a) : (b))
#define IMAGE_SIZE 28

typedef struct
{
    double delta; // for backpropagation
    double *weightsGradiantSum;
    double biasGradiantSum;
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
        layer->neurons[i].weightsGradiantSum = malloc(sizeof(double) * inputSize);
        layer->neurons[i].numWeights = inputSize;
        for (int j = 0; j < inputSize; j++)
        {
            layer->neurons[i].weights[j] = ((double)rand() / RAND_MAX * 2 - 1);
            layer->neurons[i].weightsGradiantSum[j] = 0.0;
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
        free(layer->neurons[i].weightsGradiantSum);
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

double printOutputNeuronsPercentActivate(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->outputLayer.size; i++)
    {
        printf(" (%d = %.2f%%) ", i, outputNeuronPercentActivate(nn, i));
    }

    printf("\n");
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

void layerLearnOutput(NeuralNetwork *nn, Layer *privousLayer, Layer *layer, double learnRate, Data *trainingData, double (*activationFunction)(double, int))
{
    for (int i = 0; i < layer->size; i++)
    {
        layer->neurons[i].delta = 0.0;
        layer->neurons[i].biasGradiantSum = 0.0;
        for (int j = 0; j < layer->neurons[i].numWeights; j++)
        {
            layer->neurons[i].weightsGradiantSum[j] = 0.0;
        }
    }

    addInputs(nn, trainingData->inputs);
    computeNetwork(nn);
    for (int j = 0; j < layer->size; j++)
    {
        double neuronOutput = layer->neurons[j].output;
        double targetOutput = outputNeuronExpected(j, trainingData);

        layer->neurons[j].delta = 2 * (neuronOutput - targetOutput) * activationFunction(layer->neurons[j].weightedInput, 1);

        for (int k = 0; k < layer->neurons[j].numWeights; k++)
        {
            double input = privousLayer->neurons[k].output;
            layer->neurons[j].weightsGradiantSum[k] += layer->neurons[j].delta * input;
        }

        layer->neurons[j].biasGradiantSum += layer->neurons[j].delta;
    }

    for (int i = 0; i < layer->size; i++)
    {
        for (int j = 0; j < layer->neurons[i].numWeights; j++)
        {
            layer->neurons[i].weights[j] -= layer->neurons[i].weightsGradiantSum[j] * learnRate;
        }
        layer->neurons[i].bias -= layer->neurons[i].biasGradiantSum * learnRate;
    }
}

void layerLearnIntermediate(NeuralNetwork *nn, Layer *previousLayer, Layer *layer, Layer *nextLayer, double learnRate, Data *trainingData, double (*activationFunction)(double, int))
{
    for (int i = 0; i < layer->size; i++)
    {
        layer->neurons[i].delta = 0.0;
        layer->neurons[i].biasGradiantSum = 0.0;
        for (int j = 0; j < layer->neurons[i].numWeights; j++)
        {
            layer->neurons[i].weightsGradiantSum[j] = 0.0;
        }
    }

    addInputs(nn, trainingData->inputs);
    computeNetwork(nn);
    for (int j = 0; j < layer->size; j++)
    {
        for (int l = 0; l < nextLayer->size; l++)
        {
            double weightOfNextNeuron = nextLayer->neurons[l].weights[j];
            double deltaOfNextNeuron = nextLayer->neurons[l].delta;
            layer->neurons[j].delta += weightOfNextNeuron * deltaOfNextNeuron * activationFunction(layer->neurons[j].weightedInput, 1);
        }

        for (int k = 0; k < layer->neurons[j].numWeights; k++)
        {
            double input = previousLayer->neurons[k].output;
            layer->neurons[j].weightsGradiantSum[k] += layer->neurons[j].delta * input;
        }

        layer->neurons[j].biasGradiantSum += layer->neurons[j].delta;
    }

    for (int i = 0; i < layer->size; i++)
    {
        for (int j = 0; j < layer->neurons[i].numWeights; j++)
        {
            layer->neurons[i].weights[j] -= layer->neurons[i].weightsGradiantSum[j] * learnRate;
        }
        layer->neurons[i].bias -= layer->neurons[i].biasGradiantSum * learnRate;
    }
}

void learn(NeuralNetwork *nn, double learnRate, Data *trainingData, int numData)
{
    for (int j = 0; j < numData; j++)
    {
        layerLearnOutput(nn, (nn->numHiddenLayer == 0) ? &nn->inputLayer : &nn->hiddenLayers[nn->numHiddenLayer - 1], &nn->outputLayer, learnRate, &trainingData[j], nn->activationFunction);
        for (int i = nn->numHiddenLayer - 1; i >= 0; i--)
        {
            layerLearnIntermediate(nn, (i == 0) ? &nn->inputLayer : &nn->hiddenLayers[i - 1], &nn->hiddenLayers[i], (i == nn->numHiddenLayer - 1) ? &nn->outputLayer : &nn->hiddenLayers[i + 1], learnRate, &trainingData[j], nn->activationFunction);
        }
    }
}

Data dataFromImage(char *path)
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

    return data;
}

Data *populateDataSet(int *numData, int maxEachDigit)
{
    *numData = 0;
    Data *dataSet = malloc(sizeof(Data));
    for (int i = 0; i < 10; i++)
    {
        char subdirectory[30];
        sprintf(subdirectory, "data/train/%d", i);
        DIR *dir = opendir(subdirectory);
        struct dirent *entry;
        if (dir == NULL)
        {
            fprintf(stderr, "Failed to open directory: %s\n", subdirectory);
            exit(1);
        }
        int count = 0;
        while ((entry = readdir(dir)) != NULL)
        {
            if (count >= maxEachDigit)
            {
                break;
            }
            if (entry->d_type == DT_REG)
            {
                char filepath[256];
                sprintf(filepath, "%s/%s", subdirectory, entry->d_name);
                Data newData = dataFromImage(filepath);
                *numData += 1;
                dataSet = realloc(dataSet, sizeof(Data) * (*numData));
                dataSet[*numData - 1] = newData;
            }
            count++;
        }
        closedir(dir);
    }
    return dataSet;
}

int main()
{
    int numInput = IMAGE_SIZE * IMAGE_SIZE;
    int numHiddenLayer = 1;
    int *hiddenLayerSizes = malloc(numHiddenLayer * sizeof(int));
    hiddenLayerSizes[0] = 16;
    int outputLayerSize = 10;
    double (*activationFunction)(double, int) = &sigmoid;

    NeuralNetwork network;
    initialiseNeuralNetwork(&network, numHiddenLayer, hiddenLayerSizes, outputLayerSize, numInput, activationFunction);

    int numData = 0;
    int maxEach = 50;
    Data *trainingData = populateDataSet(&numData, maxEach);

    double learnRate = 0.3;
    int learnAmmount = 500;
    int epochAmmount = 10;
    for (int i = 0; i <= learnAmmount; i++)
    {
        if (i % epochAmmount == 0)
        {
            double newCost = cost(&network, trainingData, numData);
            printf("Epoch learned %d, cost: %f \n", i, newCost);
        }
        learn(&network, learnRate, trainingData, numData);
    }

    char path[100];
    while (1)
    {
        printf("Enter the file path of the image to test (or 'q' to quit): ");
        if (scanf("%99s", path) != 1)
        {
            printf("Invalid input format. Please try again.\n");
            continue;
        }

        if (path[0] == 'q' || path[0] == 'Q')
        {
            break;
        }
        Data testData = dataFromImage(path);

        addInputs(&network, testData.inputs);
        computeNetwork(&network);
        printOutputNeuronsPercentActivate(&network);
    }
    freeNeuralNetwork(&network);
    for (int i = 0; i < numData; i++)
    {
        free(trainingData[i].inputs);
    }
    free(trainingData);
    if (numHiddenLayer > 0)
    {
        free(hiddenLayerSizes);
    }
    return 0;
}
