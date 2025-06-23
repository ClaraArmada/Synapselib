#include "library.h"

#include <cmath>
#include <numeric>
#include <functional>
#include <random>
#include <unordered_map>

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

 // Enum of every activation function

enum class e_ActivationFunctions {
    Sigmoid,
    BinaryStep,
    Linear,
    Tanh,
    ReLU,
    LeakyReLU,
    ParametricRelu,
    ELU
};

// One for 1-input functions
std::unordered_map<e_ActivationFunctions, std::function<double(double)>> simpleActivationTable;

// One for 2-input functions
std::unordered_map<e_ActivationFunctions, std::function<double(double, double)>> paramActivationTable;

// Registers the activation function
void registerActivationFunctions() {
    simpleActivationTable[e_ActivationFunctions::Sigmoid] = f_Sigmoid;
    simpleActivationTable[e_ActivationFunctions::BinaryStep] = f_BinaryStep;
    simpleActivationTable[e_ActivationFunctions::Linear] = f_Linear;
    simpleActivationTable[e_ActivationFunctions::Tanh] = f_Tanh;
    simpleActivationTable[e_ActivationFunctions::ReLU] = f_ReLU;
    simpleActivationTable[e_ActivationFunctions::LeakyReLU] = f_LeakyReLU;

    paramActivationTable[e_ActivationFunctions::ELU] = f_ELU;
    paramActivationTable[e_ActivationFunctions::ParametricRelu] = f_ParametricReLU;
}

double getActivated(double x, e_ActivationFunctions type, double alpha = 1.0) {
    if (simpleActivationTable.contains(type)) {
        return simpleActivationTable[type](x);
    }
    return paramActivationTable[type](x, alpha);
}

// perceptron //

Perceptron::Perceptron(const std::vector<double> &weights, const double bias)
    : mWeights(weights)
      , mBias(bias) {
}

void Perceptron::weightChange(const double newWeight, const int index) {
    mWeights[index] = newWeight;
}

double Perceptron::step(const std::vector<double> &inputs) const {
    return sigmoid(weightedSum(inputs));
}

void Perceptron::training(const std::vector<double> &inputs, double expectedOutput,
                          double learningRate, int maxIterations, double destinationErrorRate) {
    for (int _ = 0; _ < maxIterations; _++) {
        const double output = step(inputs);
        const double error = expectedOutput - output;

        if (abs(error) < destinationErrorRate) break;

        for (int index = 0; index < mWeights.size(); index++) {
            const double gradient = error * output * (1 - output) * inputs[index];
            mWeights[index] += learningRate * gradient;
        }
    }
}

std::vector<double> Perceptron::getWeights() const {
    return mWeights;
}

double Perceptron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Perceptron::weightedSum(const std::vector<double> &inputs) const {
    return std::inner_product(inputs.begin(), inputs.end(), mWeights.begin(), 0.0) + mBias;
}


// Neural Network //

NeuralNetwork::NeuralNetwork(const int inputLayerLength, std::vector<int> hiddenLayersLengths,
                             int outputLayerLength, std::vector<double> initialWeightsRange,
                             double bias)
    : mInputLayer(inputLayerLength) {
    int prevLayerLength = mInputLayer;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(initialWeightsRange[0], initialWeightsRange[1]);

    for (const int &LayerLength: hiddenLayersLengths) {
        std::vector<Perceptron> layer;
        for (int _ = 0; _ < LayerLength; _++) {
            std::vector<double> weights;
            for (int __ = 0; __ < prevLayerLength; __++) {
                weights.push_back(dis(gen));
            }
            layer.emplace_back(weights, bias);
        }
        mHiddenLayers.emplace_back(layer);
        prevLayerLength = LayerLength;
    }


    std::vector<Perceptron> layer = {};
    for (int _ = 0; _ < outputLayerLength; _++) {
        std::vector<double> weights;
        for (int __ = 0; __ < prevLayerLength; __++) {
            weights.push_back(dis(gen));
        }
        layer.emplace_back(weights, bias);
    }
    mOutputLayer = layer;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &inputs) {
    std::vector<double> previousOutputs = inputs;

    for (std::vector<Perceptron> &layer: mHiddenLayers) {
        std::vector<double> layerOutputs;
        for (const Perceptron &perceptron: layer) {
            layerOutputs.push_back(perceptron.step(previousOutputs));
        }
        previousOutputs = layerOutputs;
    }

    std::vector<double> outputs;
    for (Perceptron perceptron : mOutputLayer) {
        outputs.push_back(perceptron.step(previousOutputs));
    }

    return outputs;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
           NeuralNetwork::forwardPass(std::vector<double> inputs) {

}
// std::vector<std::vector<double>> activations = {inputs};
