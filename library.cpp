#include "library.h"

#include <cmath>
#include <numeric>
#include <cstdlib>
#include <random>

// perceptron //

Perceptron::Perceptron(const std::vector<double>& weights, const double bias)
    : mWeights(weights)
    , mBias(bias) {}

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
    return std::inner_product(inputs.begin(), inputs.end(), mWeights.begin(), 0.0) + mBias;
}

// Neural Network //

NeuralNetwork::NeuralNetwork(const int inputLayerLength, std::vector<int> hiddenLayersLengths,
                             int outputLayerLength, std::vector<double> initialWeightsRange,
                             double bias)
                             :mInputLayer(inputLayerLength) {
    int prevLayerLength = mInputLayer;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(initialWeightsRange[0], initialWeightsRange[1]);

    for (const int& LayerLength : hiddenLayersLengths) {
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
            weights.push_back(initialWeightsRange[0] + static_cast <float> (rand()) / static_cast <float> (RAND_MAX/(initialWeightsRange[1]-initialWeightsRange[0])));
        }
        layer.emplace_back(weights, bias);
    }
    mOutputLayer = layer;
}

