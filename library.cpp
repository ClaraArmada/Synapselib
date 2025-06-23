#include "library.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_map>

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

// Getter for activation functions
double getActivated(double x, e_ActivationFunctions type, double alpha = 1.0) {
    if (simpleActivationTable.find(type) != simpleActivationTable.end()) {
        return simpleActivationTable[type](x);
    }
    return paramActivationTable[type](x, alpha);
}



// PERCEPTRONS AND NEURONS

// base perceptron //

Perceptron::Perceptron(const std::vector<double> &weights, const double bias)
    : mWeights(weights)
      , mBias(bias) {
}

void Perceptron::weightChange(const double newWeight, const int index) {
    mWeights[index] = newWeight;
}

std::vector<double> Perceptron::getWeights() const {
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
                          double learningRate, int maxIterations, double destinationErrorRate) const {
    for (int _ = 0; _ < maxIterations; _++) {
        const double output = step(inputs);
        const double error = expectedOutput - output;

        if (abs(error) < destinationErrorRate) break;

        for (int index = 0; index < mPerceptron.getWeights().size(); index++) {
            const double gradient = error * output * (1 - output) * inputs[index];
            mPerceptron.getWeights()[index] += learningRate * gradient;
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
    ) : mInputLayer(inputLayerLength),
        mOutputLayer({}, outputLayerProperties.second, 1.0) {

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

    // Output layer
    const auto &[outputSize, outputActivation] = outputLayerProperties;
    std::vector<Perceptron> outputNeurons;
    for (int i = 0; i < outputSize; ++i) {
        std::vector<double> weights(prevLayerLength);
        for (double &w: weights) w = dis(gen);
        outputNeurons.emplace_back(weights, bias);
    }
    mOutputLayer = Layer(outputNeurons, outputActivation, 1.0);
}

double NeuralNetwork::step(const std::vector<double> &inputs, Perceptron &perceptron, e_ActivationFunctions activationFunction, double alpha) const {
    return getActivated(f_weightedSum(inputs, perceptron.getWeights(), perceptron.getBias()), activationFunction, alpha);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double> &inputs) {
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

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
           NeuralNetwork::forwardPass(std::vector<double> inputs) {
    return {{}, {}, {}};
}
// std::vector<std::vector<double>> activations = {inputs};

int main() {
    registerActivationFunctions();
    // Training data: input vectors and expected outputs
    std::vector<std::vector<double>> trainingInputs = {
        {0.0, 1.0},
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };
    std::vector<double> expectedOutputs = {1.0, 0.0, 1.0, 1.0};

    // Initial perceptron setup
    std::vector<double> initialWeights = {0.5, -0.5};
    double bias = 0.0;
    ActivatedPerceptron perceptron(initialWeights, bias, e_ActivationFunctions::Sigmoid, 1.0);

    // Train the perceptron
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (size_t i = 0; i < trainingInputs.size(); ++i) {
            perceptron.training(trainingInputs[i], expectedOutputs[i], 0.1);
        }
    }

    // Test the perceptron
    std::cout << "Testing after training:\n";
    for (size_t i = 0; i < trainingInputs.size(); ++i) {
        double output = perceptron.step(trainingInputs[i]);
        std::cout << "Input: [" << trainingInputs[i][0] << ", " << trainingInputs[i][1]
                  << "] -> Output: " << output
                  << " (Expected: " << expectedOutputs[i] << ")\n";
    }

    return 0;
}