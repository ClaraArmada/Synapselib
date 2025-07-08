#include "library.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_map>
#include <algorithm>

#include <stdexcept>
#include <cassert>


// MISC FUNCTIONS

 // Weighted Sum
 //Performs the sum of wᵢ (double) * xᵢ (double), the whole added by 2

double f_weightedSum(const std::vector<double>& inputs, const std::vector<double>& weights, const double bias = 0.0) {
    return std::inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0) + bias;
}

// ACTIVATION FUNCTIONS

 // Activation functions
double f_Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

 // binary step function
double f_BinaryStep(double x) {
    if (x < 0.0) {
        return 0.0;
    }
    return 1.0;
}

 // linear function
double f_Linear(double x) {
    return x;
}

 // hyperbolic tangent function
double f_Tanh (double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

 // rectified linear unit function
double f_ReLU (double x) {
    return std::max(0.0,x);
}

 // leaky rectified linear unit function
double f_LeakyReLU (double x) {
    return std::max(0.1 * x,x);
}

 // parametric rectified linear unit function
double f_ParametricReLU (double x, double a) {
    return std::max(a*x,x);
}

 // exponential linear unit function
double f_ELU (double x, double a) {
    if (x >= 0.0) {
        return x;
    }
    return a * (exp(x)-1);
}

// DERIVATIVE FUNCTIONS

double d_Sigmoid(double x) {
    double y = f_Sigmoid(x);
    return y * (1.0 - y);
}

double d_BinaryStep(double x) {
    return 0.0;
}

double d_Linear(double x) {
    return 1.0;
}

double d_Tanh(double x) {
    double y = f_Tanh(x);
    return 1.0 - y * y;
}

double d_ReLU(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

double d_LeakyReLU(double x) {
    return x > 0.0 ? 1.0 : 0.1;
}

double d_ParametricReLU(double x, double a) {
    return x > 0.0 ? 1.0 : a;
}

double d_ELU(double x, double a) {
    if (x >= 0.0) {
        return 1.0;
    }
    return a * exp(x);
}

// ACTIV FUNC CODE

// One for 1-input functions
std::unordered_map<e_ActivationFunctions, std::function<double(double)>> simpleActivationTable = {
    {e_ActivationFunctions::Sigmoid, f_Sigmoid},
    {e_ActivationFunctions::BinaryStep, f_BinaryStep},
    {e_ActivationFunctions::Linear, f_Linear},
    {e_ActivationFunctions::Tanh, f_Tanh},
    {e_ActivationFunctions::ReLU, f_ReLU},
    {e_ActivationFunctions::LeakyReLU, f_LeakyReLU}
};

// One for 2-input functions
std::unordered_map<e_ActivationFunctions, std::function<double(double, double)>> paramActivationTable = {
    {e_ActivationFunctions::ParametricRelu, f_ParametricReLU},
    {e_ActivationFunctions::ELU, f_ELU}
};

// One for 1-input derivatives
std::unordered_map<e_ActivationFunctions, std::function<double(double)>> simpleActivationDerivatives = {
    {e_ActivationFunctions::Sigmoid, d_Sigmoid},
    {e_ActivationFunctions::BinaryStep, d_BinaryStep},
    {e_ActivationFunctions::Linear, d_Linear},
    {e_ActivationFunctions::Tanh, d_Tanh},
    {e_ActivationFunctions::ReLU, d_ReLU},
    {e_ActivationFunctions::LeakyReLU, d_LeakyReLU}
};

// One for 2-input derivatives
std::unordered_map<e_ActivationFunctions, std::function<double(double, double)>> paramActivationDerivatives = {
    {e_ActivationFunctions::ParametricRelu, d_ParametricReLU},
    {e_ActivationFunctions::ELU, d_ELU}
};

// Getter for activation functions
double getActivated(double x, e_ActivationFunctions type, double alpha = 1.0) {
    if (simpleActivationTable.find(type) != simpleActivationTable.end()) {
        return simpleActivationTable[type](x);
    }
    return paramActivationTable[type](x, alpha);
}

double getActivatedDerivative(double x, e_ActivationFunctions type, double alpha = 1.0) {
    if (simpleActivationDerivatives.find(type) != simpleActivationDerivatives.end()) {
        return simpleActivationDerivatives[type](x);
    }
    return paramActivationDerivatives[type](x, alpha);
}

// PERCEPTRONS AND NEURONS

// base perceptron //

Perceptron::Perceptron(const std::vector<double> &weights, const double bias)
    : mWeights(weights)
      , mBias(bias) {
}

void Perceptron::weightChange(const double newWeight, const int index) {
    if (index < 0 || index >= static_cast<int>(mWeights.size())) {
        throw std::out_of_range("Weight index out of range");
    }

    mWeights[index] = newWeight;
}

std::vector<double> Perceptron::getWeights() const {
    return mWeights;
}

std::vector<double>& Perceptron::accessWeights() {
    return mWeights;
}

double Perceptron::getBias() const {
    return mBias;
}



// NEURAL NETWORKS

// Activated Perceptron

ActivatedPerceptron::ActivatedPerceptron(const std::vector<double> &weights, const double bias, e_ActivationFunctions activationFunction, double alpha)
    : mPerceptron(weights, bias)
        , mActivationFunction(activationFunction)
        , mAlpha(alpha) {
}

double ActivatedPerceptron::weightedSum(const std::vector<double>& inputs) const {
    return f_weightedSum(inputs, mPerceptron.getWeights(), mPerceptron.getBias());
}

double ActivatedPerceptron::step(const std::vector<double> &inputs) const {
    return getActivated(f_weightedSum(inputs, mPerceptron.getWeights(), mPerceptron.getBias()), mActivationFunction, mAlpha);
}

void ActivatedPerceptron::training(const std::vector<double> &inputs, double expectedOutput,
                          double learningRate, int maxIterations, double destinationErrorRate) {
    // Remove 'const' from this method signature in both .h and .cpp files

    for (int _ = 0; _ < maxIterations; _++) {
        double z = f_weightedSum(inputs, mPerceptron.getWeights(), mPerceptron.getBias());
        double output = getActivated(z, mActivationFunction, mAlpha);
        double derivative = getActivatedDerivative(z, mActivationFunction, mAlpha);
        const double error = expectedOutput - output;

        if (std::abs(error) < destinationErrorRate) break; // Use std::abs instead of abs

        // Fix the weight access issue
        std::vector<double>& weights = mPerceptron.accessWeights();
        for (size_t index = 0; index < weights.size(); index++) {
            double gradient = error * derivative * inputs[index];
            weights[index] += learningRate * gradient;
        }
    }
}


// Neural Network //

NeuralNetwork::NeuralNetwork(
    int inputLayerLength,
    const std::vector<std::pair<int, e_ActivationFunctions>>& hiddenLayersProperties,
    std::pair<int, e_ActivationFunctions> outputLayerProperties,
    const std::vector<double>& initialWeightsRange,
    double bias
    ) : mInputLayer(inputLayerLength), mOutputLayer() {  // Initialize mOutputLayer with default constructor

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(initialWeightsRange[0], initialWeightsRange[1]);

    int prevLayerLength = inputLayerLength;

    // Hidden layers
    for (const auto &[layerSize, activation]: hiddenLayersProperties) {
        std::vector<Perceptron> neurons;
        for (int i = 0; i < layerSize; ++i) {
            std::vector<double> weights(prevLayerLength);
            for (double &w: weights) w = dis(gen);
            neurons.emplace_back(weights, bias);
        }
        mHiddenLayers.emplace_back(neurons, activation, 1.0);
        prevLayerLength = layerSize;
    }

    // Output layer - Properly initialize mOutputLayer
    const auto &[outputSize, outputActivation] = outputLayerProperties;
    std::vector<Perceptron> outputNeurons;
    for (int i = 0; i < outputSize; ++i) {
        std::vector<double> weights(prevLayerLength);
        for (double &w: weights) w = dis(gen);
        outputNeurons.emplace_back(weights, bias);
    }

    // Assign the properly constructed layer to mOutputLayer
    mOutputLayer = Layer(outputNeurons, outputActivation, 1.0);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &inputs) {
    if (inputs.size() != static_cast<size_t>(mInputLayer)) {
        throw std::invalid_argument("Input size mismatch. Expected " +
                                  std::to_string(mInputLayer) +
                                  " but got " + std::to_string(inputs.size()));
    }

    std::vector<double> previousOutputs = inputs;

    for (const Layer& layer : mHiddenLayers) {
        std::vector<double> layerOutputs;
        for (const Perceptron& neuron : layer.neurons) {
            double z = f_weightedSum(previousOutputs, neuron.getWeights(), neuron.getBias());
            layerOutputs.push_back(getActivated(z, layer.activationFunction, layer.alpha));
        }
        previousOutputs = layerOutputs;
    }

    std::vector<double> outputs;
    for (const Perceptron& perceptron : mOutputLayer.neurons) {
        outputs.push_back(f_weightedSum(previousOutputs, perceptron.getWeights(), perceptron.getBias()));
    }

    return outputs;
}

std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
           NeuralNetwork::forwardPass(const std::vector<double>& inputs) {
    std::vector<std::vector<double>> activations = {inputs};
    std::vector<std::vector<double>> weightedSums;

    std::vector<double> previousOutputs = inputs;

    // Use const reference instead of copying the entire Layer
    for (const Layer& layer : mHiddenLayers) {
        std::vector<double> layerWeightedSums;
        std::vector<double> layerActivations;

        // Use const reference instead of copying the entire Perceptron
        for (const Perceptron& perceptron : layer.neurons) {
            double ws = f_weightedSum(previousOutputs, perceptron.getWeights(), perceptron.getBias());
            double act = getActivated(ws, layer.activationFunction, layer.alpha);
            layerWeightedSums.push_back(ws);
            layerActivations.push_back(act);
        }

        weightedSums.push_back(layerWeightedSums);
        activations.push_back(layerActivations);
        previousOutputs = layerActivations;
    }

    std::vector<double> outputWeightedSums;
    std::vector<double> outputActivations;

    // Use const reference instead of copying the entire Perceptron
    for (const Perceptron& perceptron : mOutputLayer.neurons) {
        double ws = f_weightedSum(previousOutputs, perceptron.getWeights(), perceptron.getBias());
        double act = getActivated(ws, mOutputLayer.activationFunction, mOutputLayer.alpha);
        outputWeightedSums.push_back(ws);
        outputActivations.push_back(act);
    }

    weightedSums.push_back(outputWeightedSums);
    activations.push_back(outputActivations);

    return std::make_tuple(outputActivations, activations, weightedSums);
}


double NeuralNetwork::lossCalculation(std::vector<double> expectedValues,
                            std::vector<double> outputValues)
{
    if (expectedValues.size() != outputValues.size()) {
        throw std::invalid_argument("Size mismatch in loss calculation");
    }

    double loss = 0.0;
    for (size_t i = 0; i < expectedValues.size(); ++i) {
        double error = expectedValues[i] - outputValues[i];
        loss += error * error;
    }

    return loss / expectedValues.size();
}



std::vector<std::vector<double>> NeuralNetwork::backPropagation(const std::vector<double>& expectedValues,
                                                                const std::vector<double>& outputActivations,
                                                                const std::vector<std::vector<double>>& activations,
                                                                const std::vector<std::vector<double>>& weightedSums) {
    std::vector<std::vector<double>> deltaAllLayers;

    std::vector<double> outputDeltas;
    for (size_t j = 0; j < mOutputLayer.neurons.size(); ++j) {
        double output = outputActivations[j];
        double delta = (expectedValues[j] - output) * output * (1.0 - output);
        outputDeltas.push_back(delta);
    }

    deltaAllLayers.push_back(outputDeltas);

    const Layer* nextLayer = &mOutputLayer;
    std::vector<double> nextDeltas = outputDeltas;

    for (int layer_idx = static_cast<int>(mHiddenLayers.size()) - 1; layer_idx >= 0; --layer_idx) {
        const Layer& layer = mHiddenLayers[layer_idx];
        const std::vector<double>& layerActivations = activations[layer_idx + 1];
        std::vector<double> layerDeltas;

        for (size_t i = 0; i < layerActivations.size(); ++i) {
            double sum_delta = 0.0;

            for (size_t j = 0; j < nextLayer->neurons.size(); ++j) {
                double weight = nextLayer->neurons[j].getWeights()[i];
                sum_delta += weight * nextDeltas[j];
            }

            double z = weightedSums[layer_idx][i];
            double derivative = getActivatedDerivative(z, layer.activationFunction, layer.alpha);
            layerDeltas.push_back(sum_delta * derivative);
        }

        deltaAllLayers.push_back(layerDeltas);
        nextLayer = &layer;
        nextDeltas = layerDeltas;
    }

    std::reverse(deltaAllLayers.begin(), deltaAllLayers.end());
    return deltaAllLayers;
}


void NeuralNetwork::weightUpdates(const std::vector<std::vector<double>>& activations, const std::vector<std::vector<double>>& deltaAllLayers, double learningRate) {// Hidden layers
    for (int layer_idx = 0; layer_idx < mHiddenLayers.size(); ++layer_idx) {
        const std::vector<double>& prevActivations = activations[layer_idx];
        const std::vector<double>& deltas = deltaAllLayers[layer_idx];
        Layer& layer = mHiddenLayers[layer_idx];

        for (size_t perceptron_idx = 0; perceptron_idx < layer.neurons.size(); ++perceptron_idx) {
            Perceptron& perceptron = layer.neurons[perceptron_idx];
            std::vector<double>& weights = perceptron.accessWeights();

            for (size_t w_idx = 0; w_idx < weights.size(); ++w_idx) {
                double gradient = deltas[perceptron_idx] * prevActivations[w_idx];
                weights[w_idx] += learningRate * gradient;
            }
        }
    }

    // Output layer
    const std::vector<double>& prevActivations = activations[activations.size() - 2];
    const std::vector<double>& deltas = deltaAllLayers.back();

    for (size_t perceptron_idx = 0; perceptron_idx < mOutputLayer.neurons.size(); ++perceptron_idx) {
        Perceptron& perceptron = mOutputLayer.neurons[perceptron_idx];
        std::vector<double>& weights = perceptron.accessWeights();

        for (size_t w_idx = 0; w_idx < weights.size(); ++w_idx) {
            double gradient = deltas[perceptron_idx] * prevActivations[w_idx];
            weights[w_idx] += learningRate * gradient;
        }
    }
}

void NeuralNetwork::training(const std::vector<double>& inputs,
                           const std::vector<double>& expectedOutput,
                           double learningRate,
                           int maxIterations,
                           int printEvery) {
    // Fix: Handle printEvery = 0 case
    if (printEvery <= 0) {
        printEvery = maxIterations; // Only print at the end
    }

    for (int epoch = 0; epoch < maxIterations; ++epoch) {
        std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> forwardResult = forwardPass(inputs);

        std::vector<double>& outputActivations = std::get<0>(forwardResult);
        std::vector<std::vector<double>>& activations = std::get<1>(forwardResult);
        std::vector<std::vector<double>>& weightedSums = std::get<2>(forwardResult);

        double loss = lossCalculation(expectedOutput, outputActivations);

        if (epoch % printEvery == 0 || epoch == maxIterations - 1 || loss < 1e-6) {
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
        }

        std::vector<std::vector<double>> deltaAllLayers = backPropagation(expectedOutput, outputActivations, activations, weightedSums);
        weightUpdates(activations, deltaAllLayers, learningRate);

        if (loss < 1e-6) {
            break;
        }
    }
}
