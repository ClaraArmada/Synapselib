#ifndef SYNAPSELIB_LIBRARY_H
#define SYNAPSELIB_LIBRARY_H

#include <memory>
#include <vector>
#include <random>


/**
 * @brief Enum of every Activation Function: Sigmoid, BinaryStep, Linear, Tanh, ReLU, LeakyReLU, ParametricRelu, ELU.
 */
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

    // Accesses the weights
    std::vector<double>& accessWeights();

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
    void training(const std::vector<double> &inputs, double expectedOutput,
                          double learningRate, int maxIterations, double destinationErrorRate);
};



// Layer struct
struct Layer {
    std::vector<Perceptron> neurons;
    e_ActivationFunctions activationFunction;
    double alpha = 1.0;

    // Add default constructor
    Layer() : activationFunction(e_ActivationFunctions::Sigmoid), alpha(1.0) {}

    Layer(std::vector<Perceptron> neurons, e_ActivationFunctions func, double alpha = 1.0)
        : neurons(std::move(neurons)), activationFunction(func), alpha(alpha) {}
};

// Add constexpr for compile-time constants
constexpr double DEFAULT_LEARNING_RATE = 0.05;
constexpr double DEFAULT_ERROR_THRESHOLD = 1e-6;
constexpr int DEFAULT_MAX_ITERATIONS = 1000;

/**
 * @brief A basic Convolution Layer
 */
class ConvolutionLayer{
    std::vector<std::vector<std::vector<std::vector<double>>>> mKernels; // vector of 3D kernels, so a 4D vector [kernel][channel][row][value]
public:
    // Constructor
    explicit ConvolutionLayer(const std::vector<int> kernelsFromCenter, int kernelsCount = 1, int channelCount = 1);

    //Performs the convolution
    std::vector<std::vector<std::vector<double>>> convolution(const std::vector<std::vector<std::vector<double>>>& inputs) const;
};

/**
* @brief Initializes a feedforward neural network with randomly assigned weights for each perceptron (neuron).
* The user specifies the number of neurons in the Input Layer, the structure of hidden layers, and the number of neurons in the output layer.
 */
class NeuralNetwork {
    std::unique_ptr<std::mt19937> m_generator;
    int mInputLayer;
    std::vector<Layer> mHiddenLayers;
    Layer mOutputLayer;
public:
    // Constructor
    explicit NeuralNetwork(int inputLayerLength,
                          const std::vector<std::pair<int, e_ActivationFunctions>>& hiddenLayersProperties,
                          std::pair<int, e_ActivationFunctions> outputLayerProperties,
                          const std::vector<double>& initialWeightsRange = {-1.0, 1.0},
                          double bias = 0.0);

    // Copy constructor
    NeuralNetwork(const NeuralNetwork& other);

    // Move constructor
    NeuralNetwork(NeuralNetwork&& other) noexcept;

    // Copy assignment
    NeuralNetwork& operator=(const NeuralNetwork& other);

    // Move assignment
    NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;

    // Destructor
    ~NeuralNetwork() = default;

    // Single epoch, computes the output(s) of (an) input(s)
    std::vector<double> predict(const std::vector<double>& inputs);

    // Performs a forward pass
    std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>>
           forwardPass(const std::vector<double>& inputs);

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

    void training(const std::vector<double>& inputs,
                        const std::vector<double>& expectedOutput,
                        double learningRate = 0.05,
                        int maxIterations = 1000,
                        int printEvery = 50);
};

/**
 * @brief A basic Convolutional Neural Network, VGG structure.
 */
class ConvolutionalNeuralNetwork {
    std::vector<std::vector<ConvolutionLayer>> mConvBlock;
    NeuralNetwork mClassifier;
public:
    ConvolutionalNeuralNetwork(const std::vector<std::vector<ConvolutionLayer>>& convolutionBlocks, NeuralNetwork classifier) : mConvBlock(convolutionBlocks), mClassifier(classifier) {}

    std::tuple<std::vector<double>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> forward(const std::vector<std::vector<std::vector<double>>>& input);

    double lossCalculation(std::vector<double> expectedValues,
                           std::vector<double> outputValues);

    std::vector<double> ConvolutionalNeuralNetwork::lossDerivative(std::vector<double> expected, std::vector<double> output);

    std::vector<std::vector<double>> ConvolutionalNeuralNetwork::Backpropagation(const std::vector<double>& expectedValues,
                                                                const std::vector<double>& outputActivations,
                                                                const std::vector<std::vector<double>>& activations,
                                                                const std::vector<std::vector<double>>& weightedSums);

    void ConvolutionalNeuralNetwork::Training(std::vector<std::vector<std::vector<double>>> inputImages,
        std::vector<double> expectedOutput,
        double learningRate,
        int maxIterations,
        int printEvery);
};

#endif //SYNAPSELIB_LIBRARY_H