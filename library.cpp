#include "library.h"

#include <cmath>
#include <numeric>

// perceptron //

Perceptron::Perceptron(const std::vector<double>& weights)
    : mWeights(weights) {}

void Perceptron::weightChange(const double newWeight, const int index) {
    mWeights[index] = newWeight;
}

double Perceptron::step(const std::vector<double>& inputs) const {
    return sigmoid(weightedSum(inputs));
}

void Perceptron::training(const std::vector<double>& inputs, double expectedOutput,
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

double Perceptron::weightedSum(const std::vector<double>& inputs) const {
    return std::inner_product(inputs.begin(), inputs.end(), mWeights.begin(), 0.0);
}

// Neural Network //

NeuralNetwork::NeuralNetwork(int inputLayerLength, std::vector<int> hiddenLayersLengths,
    int outputLayerLength, std::vector<double> initialWeightsRange) {}

