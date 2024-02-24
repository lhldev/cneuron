#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct
{
    double *weights;
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
} NeuralNetwork;

double sigmoid(double val)
{
    return 1 / (1 + exp(-val));
}

double calcOutput(Layer *previousLayer, Neuron *neuron)
{
    neuron->output = 0.0; // Initialize output
    for (int i = 0; i < previousLayer->size; i++)
    {
        neuron->output += previousLayer->neurons[i].output * neuron->weights[i];
    }
    neuron->output = sigmoid(neuron->output);
    return neuron->output;
}

void calcOutputLayer(Layer *priviousLayer, Layer *currentLayer)
{
    for (int i = 0; i < currentLayer->size; i++)
    {
        calcOutput(priviousLayer, &currentLayer->neurons[i]);
    }
}

void initialiseLayer(Layer *layer, int inputSize)
{
    for (int i = 0; i < layer->size; i++)
    {
        layer->neurons[i].weights = malloc(sizeof(double) * inputSize);
        for (int j = 0; j < inputSize; j++)
        {
            layer->neurons[i].weights[j] = ((double)rand() / RAND_MAX * 2 - 1);
        }
        layer->neurons[i].bias = 0.0;
        layer->neurons[i].output = 0.0;
    }
}

void initialiseNeuralNetwork(NeuralNetwork *nn, int numHiddenLayer, int *hiddenLayerSizes, int outputLayerSize, int numInput)
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
    initialiseLayer(&nn->outputLayer, hiddenLayerSizes[sizeof(hiddenLayerSizes) / sizeof(int) - 1]);
}

void freeLayer(Layer *layer)
{
    for (int i = 0; i < layer->size; i++)
    {
        free(layer->neurons[i].weights);
    }

    free(layer);
}

void freeNeuralNetwork(NeuralNetwork *nn)
{
    for (int i = 0; i < sizeof(nn->hiddenLayers) / sizeof(Layer); i++)
    {
        freeLayer(&nn->hiddenLayers[i]);
    }
    freeLayer(&nn->outputLayer);
    freeLayer(&nn->inputLayer);
    free(nn->hiddenLayers);
}

double cost(Layer *outputLayer, int expected)
{
    // expected number is which neuron is activated, indexed at 0
    double cost = 0;

    for (int i = 0; i < outputLayer->size; i++)
    {
        double output = outputLayer->neurons[i].output;
        if (i == expected)
        {
            cost += (output - 1) * (output - 1);
        }
        else
        {
            cost += (output - 0) * (output - 0);
        }
    }

    return cost / outputLayer->size;
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
    for (int i = 0; i < nn->numHiddenLayer; i++)
    {
        calcOutputLayer(&nn->inputLayer, &nn->hiddenLayers[i]);
    }

    calcOutputLayer(&nn->inputLayer, &nn->outputLayer);
}

void printResult(NeuralNetwork *nn)
{
    for (int i = 0; i < nn->outputLayer.size; i++)
    {
        printf("%d|", nn->outputLayer.neurons[i].output);
    }
}

int main()
{
    int numInput = 2;
    int numHiddenLayer = 1;
    int *hiddenLayerSizes = malloc(numHiddenLayer * sizeof(int));
    hiddenLayerSizes[0] = 16;
    int outputLayerSize = 10;

    NeuralNetwork network;
    initialiseNeuralNetwork(&network, numHiddenLayer, hiddenLayerSizes, outputLayerSize, numInput);

    double input[2] = {1, 2};
    addInputs(&network, input);
    computeNetwork(&network);
    printResult(&network);
    // use neural network here
    freeNeuralNetwork(&network);
    return 0;
}
