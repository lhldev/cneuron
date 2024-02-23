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
} NeuralNetwork;

double sigmoid(double val)
{
    return 1 / (1 + exp(-val));
}

double calcOutput(Layer previousLayer, Neuron *neuron)
{
    neuron->output = 0.0; // Initialize output
    for (int i = 0; i < previousLayer.size; i++)
    {
        neuron->output += previousLayer.neurons[i].output * neuron->weights[i];
    }
    neuron->output = sigmoid(neuron->output);
    return neuron->output;
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

    for (int i = 0; i < numHiddenLayer; i++)
    {
        nn->hiddenLayers[i].neurons = malloc(sizeof(Neuron) * hiddenLayerSizes[i]);
        nn->hiddenLayers[i].size = hiddenLayerSizes[i];
    }

    nn->outputLayer.neurons = malloc(sizeof(Neuron) * outputLayerSize);
    nn->outputLayer.size = outputLayerSize;

    for (int i = 0; i < sizeof(nn->hiddenLayers) / sizeof(Layer); i++)
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

int main()
{
    int numInput = 10;
    int numHiddenLayer = 1;
    int *hiddenLayerSizes = malloc(numHiddenLayer * sizeof(int));
    hiddenLayerSizes[0] = 16;
    int outputLayerSize = 10;

    NeuralNetwork network;
    initialiseNeuralNetwork(&network, numHiddenLayer, hiddenLayerSizes, outputLayerSize, numInput);

    // use neural network here
    freeNeuralNetwork(&network);
    return 0;
}
