#ifndef SYNAPSELIB_LIBRARY_H
#define SYNAPSELIB_LIBRARY_H

#include <vector>

/**
 * @brief A basic Perceptron (single-layer neuron).
 */
class Perceptron {
public:
    // Constructor
    explicit Perceptron(const std::vector<double>& weights);

    // Change the weight at a specific index
    void weightChange(double newWeight, int index);

    // Compute the output for a given input
    double step(const std::vector<double>& inputs) const;

    // Train perceptron with input data
    void training(const std::vector<double>& inputs, double expectedOutput,
                  double learningRate = 0.05, int maxIterations = 1000,
                  double destinationErrorRate = 1e-6);

    // Get the current weights
    std::vector<double> getWeights() const;

protected:
    static double sigmoid(double x);
    double weightedSum(const std::vector<double>& inputs) const;
    std::vector<double> mWeights;
};

/**
* @brief Initializes a feedforward neural network with randomly assigned weights for each perceptron (neuron).
* The user specifies the number of neurons in the Input Layer, the structure of hidden layers, and the number of neurons in the output layer.
 */
class NeuralNetwork {
public:
    // Constructor
    explicit NeuralNetwork(int inputLayerLength, std::vector<int> hiddenLayersLengths, int outputLayerLength, std::vector<double> initialWeightsRange);

    double epoch(const std::vector<double>& inputs) const;

    std::tuple<>
};

#endif //SYNAPSELIB_LIBRARY_H