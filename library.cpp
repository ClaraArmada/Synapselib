#include "library.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <unordered_map>
#include <algorithm>

#include <stdexcept>

// MISC FUNCTIONS

// Weighted Sum
//Performs the sum of wᵢ (double) * xᵢ (double), the whole added by 2

double f_weightedSum(const std::vector<double> &inputs, const std::vector<double> &weights, const double bias = 0.0) {
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
double f_Tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

// rectified linear unit function
double f_ReLU(double x) {
    return std::max(0.0, x);
}

// leaky rectified linear unit function
double f_LeakyReLU(double x) {
    return std::max(0.1 * x, x);
}

// parametric rectified linear unit function
double f_ParametricReLU(double x, double a) {
    return std::max(a * x, x);
}

// exponential linear unit function
double f_ELU(double x, double a) {
    if (x >= 0.0) {
        return x;
    }
    return a * (exp(x) - 1);
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
std::unordered_map<e_ActivationFunctions, std::function<double(double)> > simpleActivationTable = {
    {e_ActivationFunctions::Sigmoid, f_Sigmoid},
    {e_ActivationFunctions::BinaryStep, f_BinaryStep},
    {e_ActivationFunctions::Linear, f_Linear},
    {e_ActivationFunctions::Tanh, f_Tanh},
    {e_ActivationFunctions::ReLU, f_ReLU},
    {e_ActivationFunctions::LeakyReLU, f_LeakyReLU}
};

// One for 2-input functions
std::unordered_map<e_ActivationFunctions, std::function<double(double, double)> > paramActivationTable = {
    {e_ActivationFunctions::ParametricRelu, f_ParametricReLU},
    {e_ActivationFunctions::ELU, f_ELU}
};

// One for 1-input derivatives
std::unordered_map<e_ActivationFunctions, std::function<double(double)> > simpleActivationDerivatives = {
    {e_ActivationFunctions::Sigmoid, d_Sigmoid},
    {e_ActivationFunctions::BinaryStep, d_BinaryStep},
    {e_ActivationFunctions::Linear, d_Linear},
    {e_ActivationFunctions::Tanh, d_Tanh},
    {e_ActivationFunctions::ReLU, d_ReLU},
    {e_ActivationFunctions::LeakyReLU, d_LeakyReLU}
};

// One for 2-input derivatives
std::unordered_map<e_ActivationFunctions, std::function<double(double, double)> > paramActivationDerivatives = {
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


// Max Pooling Function //

std::pair<std::vector<std::vector<std::vector<double>>>,
    std::vector<std::vector<std::vector<int>>>>
maxPool2x2(const std::vector<std::vector<std::vector<double>>> &input) {
    std::vector<std::vector<std::vector<double>>> output;
    std::vector<std::vector<std::vector<int>>> maxIndices;
    int channels = input.size();
    output.resize(channels);
    maxIndices.resize(channels);

    for (int c = 0; c < channels; ++c) {
        int height = input[c].size();
        int width = input[c][0].size();
        int pooled_height = height / 2;
        int pooled_width = width / 2;

        output[c].resize(pooled_height, std::vector<double>(pooled_width, 0.0));
        maxIndices[c].resize(pooled_height, std::vector<int>(pooled_width, 0));

        for (int i = 0; i < pooled_height; ++i) {
            for (int j = 0; j < pooled_width; ++j) {
                double max_val = input[c][i * 2][j * 2];
                int max_idx = (i * 2) * width + (j * 2);
                for (int di = 0; di < 2; ++di) {
                    for (int dj = 0; dj < 2; ++dj) {
                        int row = i * 2 + di;
                        int col = j * 2 + dj;
                        if (row < height && col < width) {
                            if (input[c][row][col] > max_val) {
                                max_val = input[c][row][col];
                                max_idx = row * width + col;
                            }
                        }
                    }
                }
                output[c][i][j] = max_val;
                maxIndices[c][i][j] = max_idx;
            }
        }
    }
    return {output, maxIndices};
}

// Max Unpooling Function //

std::vector<std::vector<std::vector<std::vector<double>>>>
maxPoolBackpropagation(const std::vector<std::vector<std::vector<std::vector<double>>>> &deltaPooled,
                       const std::vector<std::vector<std::vector<int>>> &maxIndices,
                       const std::vector<std::vector<std::vector<std::vector<double>>>> &originalInput) {
    std::vector<std::vector<std::vector<std::vector<double>>>> deltaUnpooled(
        originalInput.size(), // kernels
        std::vector<std::vector<std::vector<double>>>(
            originalInput[0].size(), // channels
            std::vector<std::vector<double>>(
                originalInput[0][0].size(), // height
                std::vector<double>(originalInput[0][0][0].size(), 0.0) // width
            )
        )
    );

    for (int kernel_idx = 0; kernel_idx < deltaPooled.size(); ++kernel_idx) {
        for (int channel = 0; channel < deltaPooled[kernel_idx].size(); ++channel) {
            int height = originalInput[channel].size();
            int width = originalInput[channel][0].size();

            for (int pooledHeight = 0; pooledHeight < deltaPooled[channel].size(); ++pooledHeight) {
                for (int pooledWidth = 0; pooledWidth < deltaPooled[channel][pooledHeight].size(); ++pooledWidth) {
                    int flatIndex = maxIndices[channel][pooledHeight][pooledWidth];
                    int row = flatIndex / width;
                    int col = flatIndex % width;

                    deltaUnpooled[kernel_idx][channel][row][col] += deltaPooled[kernel_idx][channel][pooledHeight][pooledWidth];
                }
            }
        }
    }

    return deltaUnpooled;
}

// ReLU 3D //

std::vector<std::vector<std::vector<double>>> ReLU3D(const std::vector<std::vector<std::vector<double>>> &input) {
    std::vector<std::vector<std::vector<double>>> output = input;
    for (auto &channel: output) {
        for (auto &row: channel) {
            for (auto &value: row) {
                value = std::max(0.0, value);
            }
        }
    }
    return output;
}

// flattenFunction //

std::vector<double> flatten(const std::vector<std::vector<std::vector<double>>> &input) {
    std::vector<double> output;
    output.reserve(input.size() * input[0].size() * input[0][0].size());
    for (const auto &channel: input) {
        for (const auto &row: channel) {
            output.insert(output.end(), row.begin(), row.end());
        }
    }
    return output;
}

// Unflatten //

std::vector<std::vector<std::vector<double>>> unflatten(std::vector<double> flatVector, int channels, int height, int width) {
    if (flatVector.size() != static_cast<size_t>(channels * height * width)) {
        throw std::invalid_argument("Shape does not match vector siz.");
    }

    std::vector<std::vector<std::vector<double>>> output(channels,
        std::vector<std::vector<double>>(height, std::vector<double>(width)));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                output[c][h][w] = flatVector[c * height * width + h * width + w];
            }
        }
    }

    return output;
}

// Reshape //

std::vector<std::vector<std::vector<double>>> reshape(
    std::vector<double> &input, int channels, int height, int width) {

    if (input.size() != static_cast<size_t>(channels * height * width)) {
        throw std::invalid_argument("Input size does not match the specified dimensions.");
    }

    std::vector<std::vector<std::vector<double>>> output(
        channels, std::vector<std::vector<double>>(height, std::vector<double>(width)));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                output[c][h][w] = input[c * height * width + h * width + w];
            }
        }
    }

    return output;
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

std::vector<double> &Perceptron::accessWeights() {
    return mWeights;
}

double Perceptron::getBias() const {
    return mBias;
}


// NEURAL NETWORKS

// Activated Perceptron

ActivatedPerceptron::ActivatedPerceptron(const std::vector<double> &weights, const double bias,
                                         e_ActivationFunctions activationFunction, double alpha)
    : mPerceptron(weights, bias)
      , mActivationFunction(activationFunction)
      , mAlpha(alpha) {
}

double ActivatedPerceptron::weightedSum(const std::vector<double> &inputs) const {
    return f_weightedSum(inputs, mPerceptron.getWeights(), mPerceptron.getBias());
}

double ActivatedPerceptron::step(const std::vector<double> &inputs) const {
    return getActivated(f_weightedSum(inputs, mPerceptron.getWeights(), mPerceptron.getBias()), mActivationFunction,
                        mAlpha);
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
        std::vector<double> &weights = mPerceptron.accessWeights();
        for (size_t index = 0; index < weights.size(); index++) {
            double gradient = error * derivative * inputs[index];
            weights[index] += learningRate * gradient;
        }
    }
}



// Dendritic Neuron //
// will do this later



// Graph Neural Network //
// will do this later



// Spiking Neuron //
// will do this later



//



// Neural Network //

NeuralNetwork::NeuralNetwork(
    int inputLayerLength,
    const std::vector<std::pair<int, e_ActivationFunctions> > &hiddenLayersProperties,
    std::pair<int, e_ActivationFunctions> outputLayerProperties,
    const std::vector<double> &initialWeightsRange,
    double bias
) : mInputLayer(inputLayerLength), mOutputLayer() {
    // Initialize mOutputLayer with default constructor

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

    for (const Layer &layer: mHiddenLayers) {
        std::vector<double> layerOutputs;
        for (const Perceptron &neuron: layer.neurons) {
            double z = f_weightedSum(previousOutputs, neuron.getWeights(), neuron.getBias());
            layerOutputs.push_back(getActivated(z, layer.activationFunction, layer.alpha));
        }
        previousOutputs = layerOutputs;
    }

    std::vector<double> outputs;
    for (const Perceptron &perceptron: mOutputLayer.neurons) {
        outputs.push_back(f_weightedSum(previousOutputs, perceptron.getWeights(), perceptron.getBias()));
    }

    return outputs;
}

std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
NeuralNetwork::forwardPass(const std::vector<double> &inputs) {
    std::vector<std::vector<double>> activations = {inputs};
    std::vector<std::vector<double>> weightedSums;

    std::vector<double> previousOutputs = inputs;

    // Use const reference instead of copying the entire Layer
    for (const Layer &layer: mHiddenLayers) {
        std::vector<double> layerWeightedSums;
        std::vector<double> layerActivations;

        // Use const reference instead of copying the entire Perceptron
        for (const Perceptron &perceptron: layer.neurons) {
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
    for (const Perceptron &perceptron: mOutputLayer.neurons) {
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
                                      std::vector<double> outputValues) {
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


std::vector<std::vector<double>> NeuralNetwork::backPropagation(const std::vector<double> &expectedValues,
                                                                 const std::vector<double> &outputActivations,
                                                                 const std::vector<std::vector<double>> &activations,
                                                                 const std::vector<std::vector<double>> &
                                                                 weightedSums) {
    std::vector<std::vector<double>> deltaAllLayers;

    std::vector<double> outputDeltas;
    for (size_t j = 0; j < mOutputLayer.neurons.size(); ++j) {
        double output = outputActivations[j];
        double z = weightedSums.back()[j]; // Use z, the pre-activation sum
        double derivative = getActivatedDerivative(z, mOutputLayer.activationFunction, mOutputLayer.alpha);
        double delta = (expectedValues[j] - output) * derivative;
        outputDeltas.push_back(delta);
    }

    deltaAllLayers.push_back(outputDeltas);

    const Layer *nextLayer = &mOutputLayer;
    std::vector<double> nextDeltas = outputDeltas;

    for (int layer_idx = static_cast<int>(mHiddenLayers.size()) - 1; layer_idx >= 0; --layer_idx) {
        const Layer &layer = mHiddenLayers[layer_idx];
        const std::vector<double> &layerActivations = activations[layer_idx + 1];
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


void NeuralNetwork::weightUpdates(const std::vector<std::vector<double>> &activations,
                                  const std::vector<std::vector<double>> &deltaAllLayers, double learningRate) {
    // Hidden layers
    for (int layer_idx = 0; layer_idx < mHiddenLayers.size(); ++layer_idx) {
        const std::vector<double> &prevActivations = activations[layer_idx];
        const std::vector<double> &deltas = deltaAllLayers[layer_idx];
        Layer &layer = mHiddenLayers[layer_idx];

        for (size_t perceptron_idx = 0; perceptron_idx < layer.neurons.size(); ++perceptron_idx) {
            Perceptron &perceptron = layer.neurons[perceptron_idx];
            std::vector<double> &weights = perceptron.accessWeights();

            for (size_t w_idx = 0; w_idx < weights.size(); ++w_idx) {
                double gradient = deltas[perceptron_idx] * prevActivations[w_idx];
                weights[w_idx] += learningRate * gradient;
            }
        }
    }

    // Output layer
    const std::vector<double> &prevActivations = activations[activations.size() - 2];
    const std::vector<double> &deltas = deltaAllLayers.back();

    for (size_t perceptron_idx = 0; perceptron_idx < mOutputLayer.neurons.size(); ++perceptron_idx) {
        Perceptron &perceptron = mOutputLayer.neurons[perceptron_idx];
        std::vector<double> &weights = perceptron.accessWeights();

        for (size_t w_idx = 0; w_idx < weights.size(); ++w_idx) {
            double gradient = deltas[perceptron_idx] * prevActivations[w_idx];
            weights[w_idx] += learningRate * gradient;
        }
    }
}

void NeuralNetwork::training(const std::vector<double> &inputs,
                             const std::vector<double> &expectedOutput,
                             double learningRate,
                             int maxIterations,
                             int printEvery) {
    // Fix: Handle printEvery = 0 case
    if (printEvery <= 0) {
        printEvery = maxIterations; // Only print at the end
    }

    for (int epoch = 0; epoch < maxIterations; ++epoch) {
        std::tuple<std::vector<double>, std::vector<std::vector<double> >, std::vector<std::vector<double> > >
                forwardResult = forwardPass(inputs);

        std::vector<double> &outputActivations = std::get<0>(forwardResult);
        std::vector<std::vector<double> > &activations = std::get<1>(forwardResult);
        std::vector<std::vector<double> > &weightedSums = std::get<2>(forwardResult);

        double loss = lossCalculation(expectedOutput, outputActivations);

        if (epoch % printEvery == 0 || epoch == maxIterations - 1 || loss < 1e-6) {
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
        }

        std::vector<std::vector<double>> deltaAllLayers = backPropagation(
            expectedOutput, outputActivations, activations, weightedSums);
        weightUpdates(activations, deltaAllLayers, learningRate);

        if (loss < 1e-6) {
            break;
        }
    }
}



// Convolutional Layer //

ConvolutionLayer::ConvolutionLayer(const std::vector<int> kernelsFromCenter, int kernelsCount, int channelCount) {
    mKernels.resize(kernelsCount);

    for (int kernel = 0; kernel < kernelsCount; ++kernel) {
        int size = kernelsFromCenter[kernel] * 2 + 1;
        std::vector cube(channelCount, std::vector(size, std::vector(size, 0.0)));
        mKernels[kernel] = cube;
    }
}

std::vector<std::vector<std::vector<double>>> ConvolutionLayer::convolution(
    const std::vector<std::vector<std::vector<double>>> &inputs) const {
    if (inputs.empty() || inputs[0].empty() || inputs[0][0].empty()) {
        throw std::invalid_argument("Input data is empty or not properly structured.");
    }

    std::vector<std::vector<std::vector<double>>> output;
    output.resize(mKernels.size());

    int kernels = mKernels.size();
    int channels = inputs.size();
    int rows = inputs[0].size();
    int columns = inputs[0][0].size();

    for (int kernel = 0; kernel < kernels; ++kernel) {
        int kernelCenter = mKernels[kernel][0].size() / 2;

        output[kernel].resize(rows);
        for (int row = 0; row < rows; ++row) {
            output[kernel][row].resize(columns);
            for (int col = 0; col < columns; ++col) {
                double sum = 0.0;

                for (int channel = 0; channel < channels; ++channel) {
                    int kernelRows = mKernels[kernel][channel].size();
                    int kernelCols = mKernels[kernel][channel][0].size();

                    for (int kr = 0; kr < kernelRows; ++kr) {
                        for (int kc = 0; kc < kernelCols; ++kc) {
                            int inputRow = row + (kr - kernelCenter);
                            int inputCol = col + (kc - kernelCenter);

                            double inputVal = 0.0;
                            if (inputRow >= 0 && inputRow < rows &&
                                inputCol >= 0 && inputCol < columns) {
                                inputVal = inputs[channel][inputRow][inputCol];
                            }

                            sum += inputVal * mKernels[kernel][channel][kr][kc];
                        }
                    }
                }

                output[kernel][row][col] = sum;
            }
        }
    }

    return output;
}

// Convolutional Neural Network //

std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<
    double>> ConvolutionalNeuralNetwork::forward(const std::vector<std::vector<std::vector<double>>> &input) {
    auto output = input;
    mConvolutionInput.clear();
    mPreactivationZ.clear();
    mActivationOutput.clear();
    mFlattenInput.clear();
    mMaxPoolIndices.clear();

    for (auto &block: mConvBlock) {
        int blockIndex = &block - &mConvBlock[0]; // Get the index of the current block

        for (const auto &layer: block) {
            int layerIndex = &layer - &block[0]; // Get the index of the current layer

            mConvolutionInput[blockIndex][layerIndex].push_back(output); //-Cache-//
            output = layer.convolution(output);
            mPreactivationZ[blockIndex][layerIndex].push_back(output); //-Cache-//
            output = ReLU3D(output);
            mActivationOutput[blockIndex][layerIndex].push_back(output); //-Cache-//

            // T-T o em geee the operations ToT this is insane!
        }

        auto [pooledOutput, poolIndices] = maxPool2x2(output);
        output = pooledOutput;
        mMaxPoolIndices.push_back(poolIndices); //-Cache-//
    }
    mFlattenInput.push_back(output); //-Cache-//
    std::vector<double> flattenedOutput = flatten(output);

    auto [nnOutput, activations, weightedSums] = mClassifier.forwardPass(flattenedOutput);

    return std::make_tuple(nnOutput, activations, weightedSums, flattenedOutput);
}

double ConvolutionalNeuralNetwork::lossCalculation(std::vector<double> expectedValues,
                                                   std::vector<double> outputValues) {
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

std::vector<double>
ConvolutionalNeuralNetwork::lossDerivative(std::vector<double> expected, std::vector<double> output) {
    std::vector<double> dLoss(output.size());
    for (size_t i = 0; i < output.size(); ++i) {
        dLoss[i] = 2.0 * (output[i] - expected[i]) / output.size(); // gradient of MSE
    }
    return dLoss;
}

std::vector<std::vector<double>> ConvolutionalNeuralNetwork::backPropagation(const std::vector<double>& expectedOutput,
                                                                              const std::vector<double>& outputActivations,
                                                                              std::vector<double> flattenedConvolutionalOutputs,
                                                                              const std::vector<std::vector<double>>& activations,
                                                                              const std::vector<std::vector<double>>& weightedSums) {
    std::vector<double> outputError = lossDerivative(expectedOutput, outputActivations);
    std::vector<std::vector<std::vector<double>>> classifierOutputDeltas = reshape(outputError, 1, 1, mClassifier.mOutputLayer.neurons.size());

    std::vector<double> deltas;
    for (size_t i = 0; i < mClassifier.mOutputLayer.neurons.size(); ++i) {
        double z = weightedSums.back()[i];
        double derivative = getActivatedDerivative(z, mClassifier.mOutputLayer.activationFunction, mClassifier.mOutputLayer.alpha);
        deltas.push_back(classifierOutputDeltas[0][0][i] * derivative);
    }

    auto lastFeatureMap = mFlattenInput.back();
    int channels = lastFeatureMap.size();
    int height = lastFeatureMap[0].size();
    int width = lastFeatureMap[0][0].size();

    std::vector<std::vector<std::vector<double>>> unflattenedDelta = unflatten(deltas, channels, height, width);

    for (int block = mConvBlock.size() - 1; block >= 0; --block) {
        // 🏗️ backprop max pooling
        // Prepare 4D delta to match maxPoolBackpropagation signature: [kernel][channel][row][col]
        if (mActivationOutput[block].empty()) {
            throw std::runtime_error("Activation output for block is empty.");
        }
        const auto &origInput4D = mActivationOutput[block].back();

        // Broadcast the 3D unflattenedDelta across the kernel dimension
        std::vector<std::vector<std::vector<std::vector<double>>>> deltaPooled4D;
        deltaPooled4D.reserve(origInput4D.size());
        for (size_t k = 0; k < origInput4D.size(); ++k) {
            deltaPooled4D.push_back(unflattenedDelta);
        }

        std::vector<std::vector<std::vector<std::vector<double>>>> maxPoolDelta =
            maxPoolBackpropagation(deltaPooled4D, mMaxPoolIndices[block], origInput4D);

        std::vector<std::vector<std::vector<std::vector<double>>>> previousLayerDelta;

        for (int layer = mConvBlock[block].size() - 1; layer >= 0; --layer) {
            std::vector<std::vector<std::vector<std::vector<double>>>> localDelta(
                mConvBlock[block][layer].mKernels.size(), // num kernels
                std::vector<std::vector<std::vector<double>>>(
                    mConvBlock[block][layer].mKernels[0].size(), // channels
                    std::vector<std::vector<double>>(
                        mConvBlock[block][layer].mKernels[0][0].size(), // rows
                        std::vector<double>(mConvBlock[block][layer].mKernels[0][0].size(), 0.0) // cols
                    )
                )
            );
            for (int kernel_idx = 0; kernel_idx < mConvBlock[block][layer].mKernels.size(); ++kernel_idx) {
                auto &kernel = mConvBlock[block][layer].mKernels[kernel_idx];
                for (int channel = kernel.size() - 1; channel >= 0; --channel) {
                    for (int row = kernel[channel].size() - 1; row >= 0; --row) {
                        for (int value = kernel[channel][row].size() - 1; value >= 0; --value) {
                            double d_ReLU_z = getActivatedDerivative(mPreactivationZ[block][layer][kernel_idx][channel][row][value], e_ActivationFunctions::ReLU, 1.0);
                            if (layer == mConvBlock[block].size() - 1) {
                                localDelta[kernel_idx][channel][row][value] = maxPoolDelta[kernel_idx][channel][row][value] * d_ReLU_z;
                            } else {
                                localDelta[kernel_idx][channel][row][value] = previousLayerDelta[kernel_idx][channel][row][value] * d_ReLU_z; // prev layer delta doesn't exist yet
                            }
                        }
                    }
                    // computing weight/gradient step, under construction
                }
            }
            previousLayerDelta = localDelta;
        }
    }
}

void ConvolutionalNeuralNetwork::Training(std::vector<std::vector<std::vector<double>>> inputImages,
                                          std::vector<double> expectedOutput, double learningRate, int maxIterations,
                                          int printEvery) {
    for (int epoch = 0; epoch < maxIterations; ++epoch) {
        auto [nnOutput, activations, weightedSums, flattenedConvolutionalOutput] = forward(inputImages);

        double loss = lossCalculation(nnOutput, expectedOutput);

        if (epoch % printEvery == 0 || epoch == maxIterations - 1 || loss < 1e-6) {
            std::cout << "Epoch: " << epoch << " Loss: " << loss << std::endl;
        }

        std::vector<std::vector<double>> deltas = mClassifier.backPropagation(
            expectedOutput, nnOutput, activations, weightedSums);
        mClassifier.weightUpdates(activations, deltas, learningRate);

        std::vector<std::vector<double>> deltaAllLayers = backPropagation(expectedOutput, nnOutput, flattenedConvolutionalOutput, activations, weightedSums);
    }
}