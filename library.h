#ifndef SYNAPSELIB_LIBRARY_H
#define SYNAPSELIB_LIBRARY_H

#include <vector>
#include <random>

/**
 * @brief Enum of every Activation Function: Sigmoid, BinaryStep, Linear, Tanh, ReLU, LeakyReLU, ParametricRelu, ELU.
 */
enum class e_ActivationFunctions;

/**
 * @brief A basic non-functional Perceptron, to be used for other classes.
 */
class Perceptron {
    std::vector<double> mWeights;
    double mBias;
public:
    // Constructor
    explicit Perceptron(const std::vector<double>& weights, double bias = 0);

    // Change the weight at a specific index
    void weightChange(double newWeight, int index);

    // Get the current weights (vector of doubles)
    std::vector<double> getWeights() const;

    // Get the current bias (double)
    double getBias() const;
};

/**
 * @brief A basic activated Perceptron (single-layer neuron).
 */
class ActivatedPerceptron{
    Perceptron mPerceptron;
    e_ActivationFunctions mActivationFunction;
    double mAlpha;
public:
    // Constructor
    explicit ActivatedPerceptron(const std::vector<double> &weights, const double bias, e_ActivationFunctions activationFunction, double alpha = 1.0);

    //Performs the sum of wᵢ (double) * xᵢ (double)
    double weightedSum(const std::vector<double>& inputs) const;

    // Compute the output for a given input
    double step(const std::vector<double>& inputs) const;

    // Train perceptron with input (double) data
    void training(const std::vector<double>& inputs, double expectedOutput,
                  double learningRate = 0.05, int maxIterations = 1000,
                  double destinationErrorRate = 1e-6) const;
};



// Layer struct
struct Layer {
    std::vector<Perceptron> neurons;
    e_ActivationFunctions activationFunction;
    double alpha = 1.0;

    Layer(std::vector<Perceptron> neurons, e_ActivationFunctions func, double alpha = 1.0)
        : neurons(std::move(neurons)), activationFunction(func), alpha(alpha) {}
};



/**
* @brief Initializes a feedforward neural network with randomly assigned weights for each perceptron (neuron).
* The user specifies the number of neurons in the Input Layer, the structure of hidden layers, and the number of neurons in the output layer.
 */
class NeuralNetwork {
    int mInputLayer;
    std::vector<Layer> mHiddenLayers;
    Layer mOutputLayer;
public:
    // Constructor
    NeuralNetwork(int inputLayerLength,
                  const std::vector<std::pair<int, e_ActivationFunctions>>& hiddenLayersProperties,
                  std::pair<int, e_ActivationFunctions> outputLayerProperties,
                  const std::vector<double>& initialWeightsRange,
                  double bias);

    // Single epoch, computes the output(s) of (an) input(s)
    std::vector<double> predict(const std::vector<double>& inputs);

    // Performs a forward pass
    std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
        forwardPass(std::vector<double> inputs);

    // Performs a loss calculation
    // Returns the Mean Squared Error (MSE)
    double lossCalculation(std::vector<double> expectedValues,
                            std::vector<double> outputValues);

    // Performs backpropagation
    std::vector<std::vector<double>> backPropagation(const std::vector<double>& expectedValues,
                                                                    const std::vector<double>& outputActivations,
                                                                    const std::vector<std::vector<double>>& activations,
                                                                    const std::vector<std::vector<double>>& weightedSums);

    void weightUpdates(const std::vector<std::vector<double>>& activations,
                              const std::vector<std::vector<double>>& deltaAllLayers,
                              double learningRate);

    void training(std::vector<double> inputs, std::vector<double> expectedOutput, double learningRate = 0.05, int maxIterations = 1000, int printEvery = 50);
};

#endif //SYNAPSELIB_LIBRARY_H