#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define max(a, b) ((a) > (b) ? (a) : (b))

typedef struct
{
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
} NeuralNetwork;

typedef struct
{
    // temp
    double *inputs;
    double expected;
} Data;

double sigmoid(double val)
{
    return 1 / (1 + exp(-val));
}

double ReLU(double val)
{
    return max(0, val);
}

double
calcOutput(Layer *previousLayer, Neuron *neuron)
{
    neuron->output = 0.0; // Initialize output
    for (int i = 0; i < previousLayer->size; i++)
    {
        neuron->output += previousLayer->neurons[i].output * neuron->weights[i];
    }
    neuron->output = ReLU(neuron->output);
    return neuron->output;
}

void calcOutputLayer(Layer *previousLayer, Layer *currentLayer)
{
    for (int i = 0; i < currentLayer->size; i++)
    {
        calcOutput(previousLayer, &currentLayer->neurons[i]);
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
    initialiseLayer(&nn->outputLayer, (numHiddenLayer == 0) ? numInput : hiddenLayerSizes[numHiddenLayer - 1]);
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
        calcOutputLayer(&nn->inputLayer, &nn->outputLayer);
    }
    else
    {
        Layer *currLayer = &nn->inputLayer;
        for (int i = 0; i < nn->numHiddenLayer; i++)
        {
            calcOutputLayer(currLayer, &nn->hiddenLayers[i]);
            currLayer = &nn->hiddenLayers[i];
        }
        calcOutputLayer(currLayer, &nn->outputLayer);
    }
}

double cost(NeuralNetwork *nn, Data *trainingData, int numData)
{
    // expected number is which neuron is activated, indexed at 0
    double cost = 0;

    for (int i = 0; i < numData; i++)
    {
        addInputs(nn, trainingData[i].inputs);
        computeNetwork(nn);
        for (int j = 0; j < nn->outputLayer.size; j++)
        {
            double output = nn->outputLayer.neurons[j].output;
            cost += (output - trainingData[i].expected) * (output - trainingData[i].expected);
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

void layerLearn(NeuralNetwork *nn, Layer *layer, double learnRate, Data *trainingData, int numData)
{
    const double h = 0.0001;

    double originalCost = cost(nn, trainingData, numData);

    for (int i = 0; i < layer->size; i++)
    {
        for (int j = 0; j < layer->neurons[i].numWeights; j++)
        {
            layer->neurons[i].weights[j] += h;
            double costGradiantW = (cost(nn, trainingData, numData) - originalCost) / h;
            layer->neurons[i].weights[j] -= h;

            // Gradiant decent weights
            layer->neurons[i].weights[j] -= costGradiantW * learnRate;
        }

        layer->neurons[i].bias += h;
        double costGradiantB = (cost(nn, trainingData, numData) - originalCost) / h;
        layer->neurons[i].bias -= h;

        // Gradiant decent bias
        layer->neurons[i].bias -= costGradiantB * learnRate;
    }
}

void learn(NeuralNetwork *nn, double learnRate, Data *trainingData, int numData)
{
    for (int i = 0; i < nn->numHiddenLayer; i++)
    {
        layerLearn(nn, &nn->hiddenLayers[i], learnRate, trainingData, numData);
    }

    layerLearn(nn, &nn->outputLayer, learnRate, trainingData, numData);
}

// Temp test create data
Data createData(double a, double b, double expected)
{
    Data newData;
    newData.inputs = malloc(sizeof(double) * 2);
    newData.inputs[0] = a;
    newData.inputs[1] = b;
    newData.expected = expected;
    return newData;
}

int main()
{
    int numInput = 2;
    int numHiddenLayer = 1;
    int *hiddenLayerSizes = malloc(numHiddenLayer * sizeof(int));
    hiddenLayerSizes[0] = 2;
    int outputLayerSize = 1;

    NeuralNetwork network;
    initialiseNeuralNetwork(&network, numHiddenLayer, hiddenLayerSizes, outputLayerSize, numInput);

    int numData = 4;
    Data *trainingData = malloc(numData * sizeof(Data));
    // test train xor
    trainingData[0] = createData(0.0, 0.0, 0.0);
    trainingData[1] = createData(0.0, 1.0, 1.0);
    trainingData[2] = createData(1.0, 0.0, 1.0);
    trainingData[3] = createData(1.0, 1.0, 0.0);

    double learnRate = 0.008;
    int learnAmmount = 500000;
    int epochAmmount = 1000;
    for (int i = 0; i < learnAmmount; i++)
    {
        if (i % epochAmmount == 0)
        {
            double newCost = cost(&network, trainingData, numData);
            printf("Epoch learned %d, cost: %f \n", i, newCost);
        }
        learn(&network, learnRate, trainingData, numData);
    }

    double testInput[2];
    char input[3];
    while (1)
    {
        printf("Enter test input (or 'q' to quit): ");
        if (scanf("%lf %lf", &testInput[0], &testInput[1]) != 2)
        {
            scanf("%2s", input); // Read and discard extra characters
            if (input[0] == 'q' || input[0] == 'Q')
            {
                break;
            }
            else
            {
                printf("Invalid input format. Please try again.\n");
            }
        }
        else
        {
            addInputs(&network, testInput);
            computeNetwork(&network);
            printResult(&network);
            printf("\n");
        }
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
